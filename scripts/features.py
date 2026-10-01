"""
features.py -- Step 3: hand-crafted features for the manual perceptron (playground version).

A perceptron is  score = w1*f1 + w2*f2 + ... + bias,  predict X if score > 0: ONE straight
line through feature space.  So the features must put X's and O's on opposite sides of a line.

Every feature is "ink density in a region / average ink density of the whole image", measured
on the BackgroundNormalize output (ink bright, background 0), so it does not depend on stroke
thickness or exposure.  1.0 = as inky as average, 3.0 = three times inkier, 0 = empty.

Regions (data/feature_masks.png):
    centre   16x16 middle square          X crosses here, O is hollow here
    ring     annulus radius 18..28 px     O's stroke lives here, X only passes through
    diag     band along both diagonals    X's arms; O touches it at 4 points only
    corners  4 corner triangles           sounded right; measured identical for X and O (padding)
The masks themselves live in models.region_masks so the exported ManualPerceptron uses the same ones.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models import ManualPerceptron, FEATURE_NAMES, region_masks  # noqa: E402,F401

_extractor = ManualPerceptron()          # weights irrelevant here; we only call .features()


@torch.no_grad()
def features_for_split(split: str):
    """(features [N,4], labels [N] 0=O 1=X) using the browser-identical preprocessing, no augmentation."""
    from dataset import XODataset
    ds = XODataset(split, augment=False)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    return _extractor.features(X).numpy(), ds.labels.numpy()


if __name__ == "__main__":
    X, y = features_for_split("train")
    print(f"{'feature':8s} {'X median':>9s} {'X 10-90%':>14s}   {'O median':>9s} {'O 10-90%':>14s}")
    for j, name in enumerate(FEATURE_NAMES):
        fx, fo = X[y == 1, j], X[y == 0, j]
        print(f"{name:8s} {np.median(fx):9.2f} {np.percentile(fx,10):6.2f}-{np.percentile(fx,90):<6.2f}   "
              f"{np.median(fo):9.2f} {np.percentile(fo,10):6.2f}-{np.percentile(fo,90):<6.2f}")
