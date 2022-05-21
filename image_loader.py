import io
from typing import List
import requests
import torch
from PIL import Image


class ImageLoader():

    def __init__(self):
        pass

    def load_image(self, url: bytes | List[bytes]) -> List[Image.Image]:
        if isinstance(url, bytes):
            url = [url]
        images = []
        for u in url:
            r = requests.get(u)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content))
            else:
                img = None
            images.append(r.status_code, img)
        return images
