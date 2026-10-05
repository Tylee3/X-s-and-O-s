"""
dataset.py -- Step 2 (playground version): labelled crops -> tensors the class website will produce.

  1. assign_split()   train / val / test by PHOTO, never by crop
  2. XODataset        reads crops, applies augmentation (train only), then converts each crop
                      through the playground's EXACT preprocessing (contract.bgr_to_contract),
                      so the model trains on precisely what the browser will hand it.

Run directly to (re)write the `split` column and print a summary:
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

from contract import bgr_to_contract, file_to_contract

MANIFEST = Path("data/crops/manifest.csv")
LABELS = {"O": 0, "X": 1}                 # class index = position in the playground's (O, X) logits
CLASS_NAMES = ["O", "X"]

# Whole photos go to one tier so a re-shot drawing (photo6 = photo2 with flash, photo5 = photo4)
# can never sit in training AND validation.  Test = one photo never seen until the end.
TIERS = {
    "train": {"photo1", "photo2", "photo6", "photo8", "photo9", "photo10"},
    "val":   {"photo3", "photo4", "photo5", "photo7"},
    "test":  {"photo11"},
}
RAW_CACHE = 256     # crops cached in memory at this size; augmentation runs here in colour,
                    # then the contract resize to 64x64 happens LAST, like the browser


def assign_split(manifest: Path = MANIFEST) -> list[dict]:
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


def load_crop(path: Path, size: int = RAW_CACHE) -> np.ndarray:
    return cv2.resize(cv2.imread(str(path)), (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Augmentation.  Each one answers "what will be different in class?"
# Colour/geometry run on the COLOUR crop, before the contract conversion, so the
# conversion itself (the browser's job) is the last thing that happens, as in production.
# --------------------------------------------------------------------------- #
class StrokeWidthJitter:
    """Thicken or thin strokes (marker tip, pressure).  Ink is DARK in the contract tensor,
    so thickening = min over a 3x3 window = -maxpool(-x); thinning = maxpool(x)."""
    def __call__(self, x):
        r = random.random()
        if r < 0.3:
            return -F.max_pool2d(-x, 3, stride=1, padding=1)
        if r < 0.5:
            return F.max_pool2d(x, 3, stride=1, padding=1)
        return x


def stroke_contrast(x: torch.Tensor) -> float:
    """Peak 'darker than local background' value of a contract tensor [1,64,64] (0..1 scale).
    Same arithmetic as models.BackgroundNormalize, used here only to detect vanished strokes."""
    gray = ((x + 1) * 0.5).unsqueeze(0)
    dil = F.max_pool2d(gray, 11, stride=1, padding=5)
    bg = -F.max_pool2d(-dil, 11, stride=1, padding=5)
    return float(torch.clamp(bg - gray, min=0).amax())


class XODataset(Dataset):
    """One item = (tensor [1,64,64] in [-1,1] exactly as the browser makes it, label 0=O / 1=X)."""

    def __init__(self, split: str, augment: bool = False, manifest: Path = MANIFEST):
        self.rows = load_rows(split, manifest)
        self.augment = augment
        root = manifest.parent
        self.raw = [load_crop(root / r["file"]) for r in self.rows]             # BGR uint8 256x256 (augmentation source)
        self.labels = torch.tensor([LABELS[r["label"]] for r in self.rows])
        if not augment:
            # Evaluation must see EXACTLY what the website produces: the full-resolution file
            # through the playground's own loader (PIL, EXIF) and resize, no 256-px cache in between.
            self.plain = [torch.from_numpy(file_to_contract(root / r["file"])) for r in self.rows]
        # hue=0.5 -> any marker colour; brightness/contrast -> room light vs flash.
        self.colour = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.5, hue=0.5)
        self.flip = v2.RandomHorizontalFlip()
        self.stroke = StrokeWidthJitter()
        self.erase = v2.RandomErasing(p=0.3, scale=(0.01, 0.05), ratio=(0.3, 3.0), value=1.0)  # 1.0 = white: glare gap

    def __len__(self):
        return len(self.rows)

    def _augment_colour_crop(self, bgr: np.ndarray, colour_jitter: bool = True) -> np.ndarray:
        # 1. colour / brightness jitter on the colour crop (hue=0.5: any marker colour)
        if colour_jitter:
            rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1)      # CHW RGB uint8
            rgb = self.colour(rgb)
            bgr = rgb.permute(1, 2, 0).numpy()[:, :, ::-1].copy()

        # 2. ZOOM OUT + PLACE + SQUASH by extending the crop's own border outwards (replicate
        #    padding), then resizing back to the cache size.  The website never crops, so a
        #    photographed symbol can be small, off-centre and stretched (photo forced into a
        #    square).  Replicating the border keeps the background continuous: pasting the crop
        #    onto a flat colour left a rectangle edge that the model learned instead of the shape.
        frac = random.uniform(0.35, 1.0)                     # symbol width as a fraction of the frame
        aspect = random.uniform(0.75, 1.33)                  # 4:3 or 3:4 photo squashed into a square
        H, W = bgr.shape[:2]
        tx, ty = int(W / frac), int(H / frac * aspect)
        px, py = max(0, tx - W), max(0, ty - H)
        left = random.randint(0, px); top = random.randint(0, py)
        # Replicating the raw border would smear any stroke that touches the edge into a long
        # bar.  So replicate the crop's BACKGROUND (a morphological closing removes the strokes,
        # as in segment.py) and paste the real crop back in the middle: continuous surface, no bars.
        # A median blur removes thin strokes WITHOUT the brightness bias of a closing (a closing
        # is always >= the original, which left the pasted crop visibly darker than its surround).
        background = cv2.medianBlur(bgr, 31)
        canvas = cv2.copyMakeBorder(background, top, py - top, left, px - left, cv2.BORDER_REPLICATE)
        canvas[top:top + H, left:left + W] = bgr
        bgr = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)

        # 3. tilt and camera angle (rotation exposes small corners: fill them with the border colour)
        rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1)
        border = torch.cat([rgb[:, :8].flatten(1), rgb[:, -8:].flatten(1), rgb[:, :, :8].flatten(1), rgb[:, :, -8:].flatten(1)], 1)
        fill = [int(v) for v in torch.median(border, dim=1).values]
        geo = v2.Compose([
            self.flip,
            v2.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=fill),
            v2.RandomPerspective(distortion_scale=0.2, p=0.4, fill=fill),
        ])
        bgr = geo(rgb).permute(1, 2, 0).numpy()[:, :, ::-1].copy()

        # 4. The browser picks ONE source pixel per output pixel (no smoothing), so thin strokes
        #    alias differently depending on the photo's resolution.  Randomise the resolution the
        #    contract resize starts from, so the model has seen every aliasing pattern.
        s = random.randint(64, 400)
        return cv2.resize(bgr, (s, s), interpolation=cv2.INTER_AREA)

    def __getitem__(self, i):
        if not self.augment:
            return self.plain[i], self.labels[i]
        # Colour jitter can push a light marker stroke into the background: luminosity contrast
        # drops to nothing, the symbol vanishes, and we would be training on noise labelled "O".
        # If the stroke is gone (local darkness peak below ~25/255), redo without colour jitter.
        # The same happens when a strong zoom-out meets a low random source resolution: the
        # stroke blurs away before the browser-style resize.  Either way, a vanished symbol is
        # just noise with a label.  Redraw the random augmentation until the stroke survives.
        for attempt in range(8):
            bgr = self._augment_colour_crop(self.raw[i], colour_jitter=attempt < 6)
            x = torch.from_numpy(bgr_to_contract(bgr))                            # [1,64,64] in [-1,1]
            if stroke_contrast(x) >= 25 / 255:
                break
        else:
            x = torch.from_numpy(bgr_to_contract(self.raw[i]))                    # give up: plain crop
        # Stroke THINNING (max over 3x3) deletes a stroke that is already only 1-2 px wide after a
        # zoom-out -- that, not colour jitter, produced most of the blank "O" images.  Keep the
        # jittered version only if the stroke is still there.
        x2 = self.stroke(x)
        if stroke_contrast(x2) < 25 / 255:
            x2 = x
        x3 = self.erase(x2)                                                       # white patch = glare gap
        x = x3 if stroke_contrast(x3) >= 25 / 255 else x2                          # unless it wiped the symbol
        # FAINT STROKES: a light marker photographed from a distance is the #1 real-world failure
        # (faint O's were called X on the first live photos).  Pull the whole image towards white
        # by a random factor so the stroke's contrast drops to 15-100% of what it was; the
        # in-model BackgroundNormalize then has to recover the shape from a weak signal, as it
        # will in class.  Keep the guard so the stroke stays detectable at all.
        if random.random() < 0.5:
            k = random.uniform(0.15, 1.0)
            x_faint = (1.0 - (1.0 - x) * k).clamp(-1, 1)                          # white = +1 stays, ink moves towards white
            if stroke_contrast(x_faint) >= 12 / 255:
                x = x_faint
        x = (x + 0.03 * torch.randn_like(x)).clamp(-1, 1)                        # sensor noise / board texture
        return x, self.labels[i]


def summary(rows):
    from collections import Counter
    print(f"{'split':6s} {'photos':42s} {'X':>4s} {'O':>4s} {'total':>6s}")
    for tier in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == tier]
        c = Counter(r["label"] for r in sub)
        print(f"{tier:6s} {', '.join(sorted(TIERS[tier], key=lambda s: int(s[5:]))):42s} {c['X']:4d} {c['O']:4d} {len(sub):6d}")
    print(f"excluded (unlabeled / junk): {sum(1 for r in rows if r['split'] == '')}")


if __name__ == "__main__":
    summary(assign_split())
