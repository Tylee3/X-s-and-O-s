"""
preview_preproc.py -- SEE what the model gets, instead of trusting it.    -> data/preview_preproc.jpg

Each row = one random training crop:
    raw crop | browser input (contract gray 64x64) | model's view after BackgroundNormalize | 6 augmented browser inputs
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import XODataset, CLASS_NAMES   # noqa: E402
from models import BackgroundNormalize       # noqa: E402

N_ROWS, N_AUG, TILE = 8, 6, 96


def tile(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    g = cv2.resize(g, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def main():
    random.seed(0); torch.manual_seed(0)
    aug = XODataset("train", augment=True)
    plain = XODataset("train", augment=False)
    norm = BackgroundNormalize()
    heads = ["raw", "browser 64x64", "model view"] + [f"aug{i + 1}" for i in range(N_AUG)]
    sheet = np.full((N_ROWS * TILE + 20, len(heads) * TILE, 3), 255, np.uint8)
    for c, h in enumerate(heads):
        cv2.putText(sheet, h, (c * TILE + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    for r, i in enumerate(random.sample(range(len(aug)), N_ROWS)):
        y = 20 + r * TILE
        raw = cv2.resize(aug.raw[i], (TILE, TILE))
        cv2.putText(raw, CLASS_NAMES[int(aug.labels[i])], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        x = plain[i][0]
        tiles = [raw, tile(x[0].numpy(), -1, 1), tile(norm(x.unsqueeze(0))[0, 0].numpy(), 0, 1)]
        tiles += [tile(aug[i][0][0].numpy(), -1, 1) for _ in range(N_AUG)]
        for c, t in enumerate(tiles):
            sheet[y:y + TILE, c * TILE:(c + 1) * TILE] = t
    out = Path("data/preview_preproc.jpg")
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", out)


if __name__ == "__main__":
    main()
