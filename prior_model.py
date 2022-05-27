import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(),device=device) * self.std + self.mean

class TimeEmbedding2(nn.Module):
    def __init__(self, size: int = 512):
        super(TimeEmbedding2, self).__init__()
        self.size = size
        inv_freq = torch.exp(
            torch.arange(0, size, 2, dtype=torch.float32) * (-math.log(10000) / size)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.unsqueeze(-1)
        x = x * self.inv_freq.unsqueeze(0)
        x = x.transpose(1, 2)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input: torch.Tensor):
        time_embedding = torch.zeros_like(input)
        time_embedding[:, :, 0::2] = torch.sin(input[:, :, 0::2] * self.inv_freq)
        time_embedding[:, :, 1::2] = torch.cos(input[:, :, 1::2] * self.inv_freq)
        return time_embedding

def noise_like(shape, noise_fn, device, repeat=False):
    if repeat:
        resid = [1] * (len(shape) - 1)
        shape_one = (1, *shape[1:])

        return noise_fn(*shape_one, device=device).repeat(shape[0], *resid)

    else:
        return noise_fn(*shape, device=device)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

class GaussianDiffusion(nn.Module):
    def __init__(self, betas):
        super().__init__()

        betas = betas.type(torch.float64)
        timesteps = betas.shape[0]
        self.num_timesteps = int(timesteps)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))
        self.register("posterior_variance", posterior_variance)
        self.register(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)),
        )
        self.register(
            "posterior_mean_coef2",
            ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)),
        )

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    def p_loss(self, model, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        x_noise = self.q_sample(x_0, t, noise)
        x_recon = model(x_noise, t)

        return F.mse_loss(x_recon, noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_0, x_t, t):
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, var, log_var_clipped

    def p_mean_variance(self, model, x, t, clip_denoised):
        x_recon = self.predict_start_from_noise(x, t, noise=model(x, t))

        if clip_denoised:
            x_recon = x_recon.clamp(min=-1, max=1)

        mean, var, log_var = self.q_posterior(x_recon, x, t)

        return mean, var, log_var

    def p_sample(self, model, x, t, noise_fn, clip_denoised=True, repeat_noise=False):
        mean, _, log_var = self.p_mean_variance(model, x, t, clip_denoised)
        noise = noise_like(x.shape, noise_fn, x.device, repeat_noise)
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)

        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, noise_fn=torch.randn):
        img = noise_fn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
            )

        return img


class PriorModel(nn.Module):

    def __init__(self):
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True, activation="relu", device=device)
        self.prior = torch.nn.TransformerDecoder(transformer_decoder_layer, num_layers=2).to(device)
        self.time = TimeEmbedding(512)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        x = self.time(x)
        x = self.prior(x,mem)
        return x

