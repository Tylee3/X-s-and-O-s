"""Reference image preprocessing for browser parity and optional export samples."""
import numpy as np
from PIL import Image, ImageOps
try:
    from pillow_heif import register_heif_opener
except ImportError:  # HEIC support is optional for our use
    def register_heif_opener():
        pass
import torch

register_heif_opener()
CONTRACT_VERSION = 'ox-gray64-v1'
CLASSES = ['O', 'X']


def load_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGBA')


def preprocess_rgba(rgba):
    rgba = np.asarray(rgba, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or not all(rgba.shape[:2]):
        raise ValueError('Expected nonempty H×W×4 RGBA pixels')
    h, w = rgba.shape[:2]
    xs = np.floor((np.arange(64) + .5) * w / 64).astype(int)
    ys = np.floor((np.arange(64) + .5) * h / 64).astype(int)
    pixels = rgba[ys[:, None], xs[None, :]].astype(np.int32)
    alpha = pixels[..., 3:4]
    rgb = (pixels[..., :3] * alpha + 255 * (255 - alpha) + 127) // 255
    gray = (299 * rgb[..., 0] + 587 * rgb[..., 1] + 114 * rgb[..., 2] + 500) // 1000
    # Float64 operation followed by float32 matches JavaScript's Number -> Float32Array.
    normalized = (gray.astype(np.float64) / 127.5 - 1).astype(np.float32)
    return gray.astype(np.uint8), normalized[None, :, :]


def preprocess(image):
    return torch.from_numpy(preprocess_rgba(np.array(image.convert('RGBA')))[1])
