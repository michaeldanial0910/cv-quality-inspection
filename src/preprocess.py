"""The single preprocessing function used everywhere a model sees an image.

The Colab notebook copies this exact code (make_colab_notebook.py inlines this file), so
training, evaluation, the API and the app all feed the network identical tensors.
Augmentation is NOT here: it is applied only to training images, inside the notebook.
"""
import numpy as np
from PIL import Image

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_gray_image(img) -> Image.Image:
    """Path / PIL image / array -> grayscale PIL image (the camera is monochrome)."""
    if isinstance(img, (str, bytes)) or hasattr(img, "__fspath__"):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.convert("L")


def to_input(img) -> np.ndarray:
    """-> float32 (3, 224, 224): resize (bilinear), scale to [0, 1], repeat to 3 channels, ImageNet-normalise."""
    g = to_gray_image(img).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    x = np.asarray(g, dtype=np.float32) / 255.0
    x = np.repeat(x[None], 3, axis=0)
    return ((x - MEAN[:, None, None]) / STD[:, None, None]).astype(np.float32)
