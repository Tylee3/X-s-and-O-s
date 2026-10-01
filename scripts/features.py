"""
features.py -- Step 3: hand-crafted features for the manual perceptron.

A perceptron is  score = w1*f1 + w2*f2 + ... + bias,  predict X if score > 0.
That is ONE straight line through feature space, so the features must be chosen
so that X's and O's sit on opposite sides of some line.  Every feature below is
"ink density in a region of the 32x32 image, divided by the average ink density
of the whole image".  Dividing by the average makes the number independent of
stroke thickness: 1.0 means "as inky as the picture on average", 3.0 means
"three times inkier than average", 0 means empty.

Regions (see data/feature_masks.png):
    centre   8x8 square in the middle      X crosses here, O is hollow here
    ring     annulus radius 9..14 px       O's stroke lives here, X only passes through
    diag     band along both diagonals     X's arms, O touches at 4 points only
    corners  4 triangles in the corners    X's arm tips reach them, O never does
"""
import numpy as np

SIZE = 32
FEATURE_NAMES = ["centre", "ring", "diag", "corners"]


def _masks(n: int = SIZE) -> dict:
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    r = np.hypot(yy - c, xx - c)
    m = {}
    m["centre"] = (np.abs(yy - c) <= 4) & (np.abs(xx - c) <= 4)          # 8x8 middle square
    m["ring"] = (r >= 9) & (r <= 14)
    m["diag"] = (np.abs(yy - xx) <= 2) | (np.abs(yy + xx - (n - 1)) <= 2)
    m["corners"] = ((yy + xx) < 11) | ((yy + xx) > 2 * (n - 1) - 11) | \
                   ((yy - xx) > 20) | ((xx - yy) > 20)
    return m


MASKS = _masks()


def extract(img: np.ndarray) -> np.ndarray:
    """img: 32x32 float ink image (0 = background, ~1 = stroke) -> 4 feature values."""
    avg = img.mean() + 1e-6
    return np.array([img[MASKS[k]].mean() / avg for k in FEATURE_NAMES], dtype=np.float32)


def features_for_split(split: str, size: int = SIZE):
    """Returns (features [N,4], labels [N] with 0=O 1=X) using the Step-2 preprocessing, no augmentation."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import XODataset
    ds = XODataset(split, size=size, augment=False)
    X = np.stack([extract(ds[i][0][0].numpy()) for i in range(len(ds))])
    y = ds.labels.numpy()
    return X, y


if __name__ == "__main__":
    X, y = features_for_split("train")
    print(f"{'feature':8s} {'X median':>9s} {'X 10-90%':>14s}   {'O median':>9s} {'O 10-90%':>14s}")
    for j, name in enumerate(FEATURE_NAMES):
        fx, fo = X[y == 1, j], X[y == 0, j]
        print(f"{name:8s} {np.median(fx):9.2f} {np.percentile(fx,10):6.2f}-{np.percentile(fx,90):<6.2f}   "
              f"{np.median(fo):9.2f} {np.percentile(fo,10):6.2f}-{np.percentile(fo,90):<6.2f}")
