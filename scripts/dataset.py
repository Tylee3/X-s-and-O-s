"""
dataset.py -- Step 2: turn the labelled crops into something PyTorch can train on.

Three jobs live here:
  1. assign_split()   decide which crops are train / val / test  (by PHOTO, never by crop)
  2. to_ink()         preprocessing: colour photo crop -> lighting- and colour-invariant ink image
  3. XODataset        a torch Dataset that applies preprocessing (+ augmentation for training)

Run it directly to (re)write the `split` column into the manifest and print a summary:
    python scripts/dataset.py
"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

MANIFEST = Path("data/crops/manifest.csv")
LABELS = {"O": 0, "X": 1}                 # class index the network predicts
CLASS_NAMES = ["O", "X"]

# --------------------------------------------------------------------------- #
# 1. The split.  Whole photos go to one tier so that the same drawing can never
#    be in training AND validation (photo6 is photo2 re-shot with flash, photo5 is
#    photo4 re-shot).  Test = one photo the model never sees until the very end.
# --------------------------------------------------------------------------- #
TIERS = {
    "train": {"photo1", "photo2", "photo6", "photo8", "photo9", "photo10"},
    "val":   {"photo3", "photo4", "photo5", "photo7"},
    "test":  {"photo11"},
}
RAW_CACHE = 96      # crops are cached in memory at this size (px); augmentations run here,
                    # then the final resize to the model size happens last (less aliasing)


def assign_split(manifest: Path = MANIFEST) -> list[dict]:
    """Add/refresh a `split` column.  Rows without an X/O label get split='' (excluded)."""
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        rows, fields = list(reader), list(reader.fieldnames)
    if "split" not in fields:
        fields.append("split")
    photo_to_tier = {p: t for t, ps in TIERS.items() for p in ps}
    for r in rows:
        r["split"] = photo_to_tier.get(r["photo"], "") if r["label"] in LABELS else ""
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def load_rows(split: str, manifest: Path = MANIFEST) -> list[dict]:
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if "split" not in rows[0]:
        raise SystemExit("manifest has no `split` column yet: run  python scripts/dataset.py")
    return [r for r in rows if r["split"] == split]


# --------------------------------------------------------------------------- #
# 2. Preprocessing.  Same idea as segment.py, now applied to one crop:
#      ink   = min(R, G, B)            any marker colour is dark in at least one channel
#      bg    = closing(ink)            what the crop would look like with no stroke
#      dark  = bg - ink                "how much darker than the surroundings" -> lighting-free
#      out   = dark / max(dark)        strongest stroke = 1.0 regardless of marker/exposure
#    The result is a float image, background 0, stroke bright.  Bright-on-black is also
#    convenient later: rotating or shifting the image pads with 0 = "more background".
# --------------------------------------------------------------------------- #
def to_ink(bgr: np.ndarray, binarize: bool = False) -> np.ndarray:
    ink = bgr.min(axis=2)
    k = max(7, (bgr.shape[0] // 6) | 1)                          # kernel wider than any stroke
    bg = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    dark = cv2.subtract(bg, ink).astype(np.float32)
    dark = cv2.GaussianBlur(dark, (3, 3), 0)                      # kill single-pixel JPEG noise
    peak = max(np.percentile(dark, 99.5), 10.0)                   # floor: 10 faint flash-hotspot crops in photo6 sit at 8-25; below 10 is noise
    out = np.clip(dark / peak, 0.0, 1.0)
    if binarize:
        out = (out > 0.4).astype(np.float32)
    return out


def load_crop(path: Path, size: int = RAW_CACHE) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# 3. Augmentation.  Each transform answers one "what will be different in class?"
# --------------------------------------------------------------------------- #
class StrokeWidthJitter:
    """Randomly thicken or thin strokes (marker tip / pressure unknown in class).
    Dilation = max over a 3x3 window; erosion = min over a 3x3 window."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r = random.random()
        if r < 0.3:
            return F.max_pool2d(x, 3, stride=1, padding=1)
        if r < 0.5:
            return -F.max_pool2d(-x, 3, stride=1, padding=1)
        return x


class AddNoise:
    """Small Gaussian noise: sensor noise / whiteboard texture."""
    def __init__(self, sigma=0.03):
        self.sigma = sigma
    def __call__(self, x):
        return (x + self.sigma * torch.randn_like(x)).clamp(0, 1)


def photometric_aug():
    """Applied to the RAW colour crop, before to_ink.  hue=0.5 means any marker colour;
    brightness/contrast mimic room light vs flash.  If to_ink really cancels these, the
    model never sees the difference -- this transform is there to make sure of it."""
    return v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.6, hue=0.5)


def geometric_aug():
    """Applied to the ink image.  Rotation/flip: a blindfolded hand tilts shapes, and X / O
    are both symmetric so flips cost nothing.  Translate/scale: box padding and drawn size
    vary.  Perspective: camera not square to the board.  Erasing: glare breaking a stroke."""
    return v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), fill=0),
        v2.RandomPerspective(distortion_scale=0.2, p=0.4, fill=0),
        StrokeWidthJitter(),
        v2.RandomErasing(p=0.3, scale=(0.01, 0.05), ratio=(0.3, 3.0), value=0),
    ])


class XODataset(Dataset):
    """One item = (image tensor [1, size, size] in 0..1, label int).  Crops are read once
    and cached in memory at RAW_CACHE px (652 crops * 96*96*3 bytes ~ 18 MB)."""

    def __init__(self, split: str, size: int = 32, binarize: bool = False, augment: bool = False,
                 manifest: Path = MANIFEST):
        self.rows = load_rows(split, manifest)
        self.size, self.binarize, self.augment = size, binarize, augment
        root = manifest.parent
        self.raw = [load_crop(root / r["file"]) for r in self.rows]            # BGR uint8 96x96
        self.labels = torch.tensor([LABELS[r["label"]] for r in self.rows])
        self.photo_aug = photometric_aug()
        self.geo_aug = geometric_aug()
        self.noise = AddNoise()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        bgr = self.raw[i]
        if self.augment:
            rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1)      # HWC BGR -> CHW RGB
            rgb = self.photo_aug(rgb)
            bgr = rgb.permute(1, 2, 0).numpy()[:, :, ::-1].copy()
        x = torch.from_numpy(to_ink(bgr, self.binarize)).unsqueeze(0)          # [1, 96, 96]
        if self.augment:
            x = self.geo_aug(x)
        x = F.interpolate(x.unsqueeze(0), size=(self.size, self.size), mode="area").squeeze(0)
        if self.augment:
            x = self.noise(x)
        if self.binarize:
            x = (x > 0.4).float()                                              # re-binarize after resize
        return x, self.labels[i]


def summary(rows):
    from collections import Counter
    print(f"{'split':6s} {'photos':40s} {'X':>4s} {'O':>4s} {'total':>6s}")
    for tier in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == tier]
        c = Counter(r["label"] for r in sub)
        print(f"{tier:6s} {', '.join(sorted(TIERS[tier], key=lambda s: int(s[5:]))):40s} {c['X']:4d} {c['O']:4d} {len(sub):6d}")
    excluded = sum(1 for r in rows if r["split"] == "")
    print(f"excluded (unlabeled / junk): {excluded}")


if __name__ == "__main__":
    summary(assign_split())
