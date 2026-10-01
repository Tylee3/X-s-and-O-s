"""
contract.py -- the playground's fixed preprocessing, wrapped for our pipeline.

The class website (supertweety/model-playground) does its OWN preprocessing in the
browser before our model sees anything, and we cannot change it:
    1. resize to 64x64 by picking one source pixel per output pixel (no smoothing)
    2. composite onto white (for transparent PNGs)
    3. grayscale by weighted luminosity  gray = (299 R + 587 G + 114 B) / 1000
    4. scale to float32 in [-1, +1]:  black = -1, white = +1
Our model receives that tensor, shape [1, 1, 64, 64], and must return [1, 2] logits (O, X).

So training data must be prepared with EXACTLY the same steps.  We vendor the
playground's own Python implementation (scripts/playground_lib/common.py) and call it,
rather than re-implementing it, so a crop looks identical to the model here and in
the browser.

CONSEQUENCE worth saying out loud: colour is gone before our model runs.  Our Step-1
trick "take the darkest of R,G,B so light-coloured markers keep contrast" is impossible
behind this interface.  What survives is the OTHER Step-1 idea, subtracting the local
background to cancel lighting, because that needs only brightness.  We do it INSIDE the
model (models.BackgroundNormalize) so it is part of the exported file.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from playground_lib.common import preprocess_rgba, CONTRACT_VERSION, CLASSES  # noqa: E402,F401

SIZE = 64


def bgr_to_contract(bgr: np.ndarray) -> np.ndarray:
    """OpenCV BGR uint8 crop (any size) -> float32 [1, 64, 64] in [-1, 1], exactly as the browser would."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    _gray_uint8, normalized = preprocess_rgba(rgba)
    return normalized


def file_to_contract(path) -> np.ndarray:
    """Image file -> float32 [1, 64, 64], via PIL with EXIF rotation, like the website's photo upload."""
    from playground_lib.common import load_image
    return preprocess_rgba(np.array(load_image(path)))[1]
