"""
preview_preproc.py -- SEE the preprocessing and augmentation instead of trusting it.

    python scripts/preview_preproc.py            -> data/preview_preproc.jpg

Each row is one random training crop:
    raw | ink (what the model sees) | binarized ink | 6 augmented versions of it
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import XODataset, to_ink, CLASS_NAMES  # noqa: E402

N_ROWS, N_AUG, TILE, SIZE = 8, 6, 96, 32


def tile_gray(x: np.ndarray) -> np.ndarray:
    """0..1 float (any size) -> TILE x TILE BGR, nearest-neighbour so the 32x32 pixels stay visible."""
    g = cv2.resize((x * 255).astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def main():
    random.seed(0); torch.manual_seed(0)
    ds = XODataset("train", size=SIZE, augment=True)
    plain = XODataset("train", size=SIZE, augment=False)
    cols = 3 + N_AUG
    sheet = np.full((N_ROWS * TILE + 20, cols * TILE, 3), 255, np.uint8)
    heads = ["raw", "ink", "binary"] + [f"aug{i+1}" for i in range(N_AUG)]
    for c, h in enumerate(heads):
        cv2.putText(sheet, h, (c * TILE + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    idxs = random.sample(range(len(ds)), N_ROWS)
    for r, i in enumerate(idxs):
        y = 20 + r * TILE
        raw = cv2.resize(ds.raw[i], (TILE, TILE))
        cv2.putText(raw, CLASS_NAMES[int(ds.labels[i])], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles = [raw, tile_gray(plain[i][0][0].numpy()), tile_gray((to_ink(ds.raw[i], binarize=True)))]
        tiles += [tile_gray(ds[i][0][0].numpy()) for _ in range(N_AUG)]
        for c, t in enumerate(tiles):
            sheet[y:y + TILE, c * TILE:(c + 1) * TILE] = t
    out = Path("data/preview_preproc.jpg")
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", out)


if __name__ == "__main__":
    main()
