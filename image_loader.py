import io
from typing import List
import requests
from PIL import Image
import numpy as np
import h5py

def load_image(url: bytes | List[bytes]) -> List[Image.Image]:
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


def check_urls(urls: np.ndarray) -> np.ndarray:
    """
    Check if the urls are valid
    :param urls:
    :return:
    """
    healthy_urls = np.zeros(urls.shape, dtype=bool)
    for idx, url in enumerate(urls):
        r = requests.get(url)
        if r.status_code == 200:
            healthy_urls[idx] = True
        print(f"{idx}/{urls.shape[0]}", end="\r")
    return healthy_urls
    

if __name__ == "__main__":
    with h5py.File("data/eee443_project_dataset_train.h5", "r") as f:
        print("Keys: %s" % f.keys())
        train_url = np.array(f["train_url"])
    with h5py.File("data/eee443_project_dataset_test.h5", "r") as f:
        print("Keys: %s" % f.keys())
        test_url = np.array(f["test_url"])

        healty_train_urls = check_urls(train_url)
        healty_test_urls = check_urls(test_url)
        with open("data/healty_train_urls.npy", "wb") as f:
            np.save(f, healty_train_urls)
        with open("data/healty_test_urls.npy", "wb") as f:
            np.save(f, healty_test_urls)