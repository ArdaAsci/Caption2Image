import io
import torch
import requests
from PIL import Image
import numpy as np
import h5py
import clip

if __name__ == "__main__":
    with h5py.File("data/eee443_project_dataset_train.h5", "r") as f:
        print("Keys: %s" % f.keys())
        train_url = np.array(f["train_url"])
    train_health = np.load("data/healty_train_urls.npy")
    device = "cpu"
    model, preprocessor = clip.load("ViT-B/32", device="cpu")
    IMS_SIZE = train_url.shape[0]

    train_image_features = torch.empty((IMS_SIZE,512), device=device)
    with torch.no_grad():
        for i in range(IMS_SIZE):
            if not train_health[i]:
                train_image_features[i] = torch.zeros(512, device=device)
                continue
            r = requests.get(train_url[i])
            image = Image.open(io.BytesIO(r.content))
            image = preprocessor(image).unsqueeze(0).to(device)
            train_image_features[i] = model.encode_image(image).float()
            print(f"Encoded {i}/{IMS_SIZE} images", end="\r")
    torch.save(train_image_features, "data/train_image_features.pt")