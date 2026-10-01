"""
perceptron_manual.py -- Classifier 1: ONE perceptron with weights chosen BY HAND (no training).

    score = WEIGHTS . features + BIAS          predict X if score > 0, else O

Edit WEIGHTS / BIAS below, re-run, read the accuracy.  Features: see features.py.
    python scripts/perceptron_manual.py --show-mistakes
The same WEIGHTS / BIAS are what export_playground.py bakes into perceptron.onnx.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from features import features_for_split, FEATURE_NAMES   # noqa: E402
from models import ManualPerceptron, region_masks, BackgroundNormalize  # noqa: E402

# ----------------------------------------------------------------------------- #
# HAND-CHOSEN.  Positive weight = "more of this pushes towards X".  BIAS = -threshold.
# ----------------------------------------------------------------------------- #
WEIGHTS = {
    "centre":  1.0,     # X crosses in the middle -> high centre density
    "ring":   -1.0,     # O's stroke lives in the ring -> high ring density means O
    "diag":    0.0,
    "corners": 0.0,     # measured: identical for X and O (crop padding), cannot help
}
BIAS = -0.3


def make_model() -> ManualPerceptron:
    return ManualPerceptron(weights=tuple(WEIGHTS[k] for k in FEATURE_NAMES), bias=BIAS).eval()


def predict(F: np.ndarray) -> np.ndarray:
    w = np.array([WEIGHTS[k] for k in FEATURE_NAMES], dtype=np.float32)
    return (F @ w + BIAS > 0).astype(int)


def report(name, F, y):
    p = predict(F)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    print(f"{name:5s} acc={(p == y).mean():.3f}  (always-guess-majority {max(y.mean(), 1 - y.mean()):.3f}, coin flip 0.5)")
    print(f"      X right {tp:3d}   X called O {fn:3d}   O right {tn:3d}   O called X {fp:3d}")
    return p


def figure(Ftr, ytr, Fva, yva):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from dataset import XODataset
    ds = XODataset("train"); norm = BackgroundNormalize(); masks = region_masks()
    ix = int(np.where(ds.labels.numpy() == 1)[0][0]); io = int(np.where(ds.labels.numpy() == 0)[0][0])
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for row, (i, nm) in enumerate([(ix, "X"), (io, "O")]):
        x = ds[i][0].unsqueeze(0); img = norm(x)[0, 0].numpy(); f = make_model().features(x)[0].numpy()
        for col, k in enumerate(FEATURE_NAMES):
            ax = axes[row, col]; m = masks[k].numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.35, vmin=0, vmax=1)
            ax.set_title(f"{nm}: {k} = {f[col]:.2f}"); ax.axis("off")
    fig.suptitle("Feature regions (orange) on the model's view of one X and one O; value = region density / average density")
    fig.tight_layout(); fig.savefig("data/feature_masks.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for col, k in enumerate(FEATURE_NAMES):
        ax = axes[col]
        ax.hist(Ftr[ytr == 1, col], bins=30, alpha=0.6, label="X", color="#1baf7a")
        ax.hist(Ftr[ytr == 0, col], bins=30, alpha=0.6, label="O", color="#2a78d6")
        ax.set_title(f"{k} (train)"); ax.legend(frameon=False)
    ax = axes[4]
    for F, y, mk, lab in [(Ftr, ytr, "o", "train"), (Fva, yva, "^", "val")]:
        ax.scatter(F[y == 1, 0], F[y == 1, 1], c="#1baf7a", marker=mk, s=14, label=f"X {lab}")
        ax.scatter(F[y == 0, 0], F[y == 0, 1], c="#2a78d6", marker=mk, s=14, label=f"O {lab}")
    wc, wr = WEIGHTS["centre"], WEIGHTS["ring"]
    if wr != 0:
        xs = np.linspace(0, Ftr[:, 0].max(), 50); ax.plot(xs, -(wc * xs + BIAS) / wr, "k--", label="score = 0")
    ax.set_xlabel("centre"); ax.set_ylabel("ring"); ax.set_ylim(0, 3); ax.set_title("decision line"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig("data/perceptron_features.png", dpi=110); plt.close(fig)
    print("figures -> data/feature_masks.png data/perceptron_features.png")


def mistakes_sheet(split, F, y, p, out):
    from dataset import XODataset, CLASS_NAMES
    ds = XODataset(split); bad = np.where(p != y)[0]
    if len(bad) == 0:
        print("no mistakes on", split); return
    tile, cols = 96, 10
    sheet = np.full((int(np.ceil(len(bad) / cols)) * tile, cols * tile, 3), 255, np.uint8)
    for n, i in enumerate(bad):
        r, c = divmod(n, cols); t = cv2.resize(ds.raw[i], (tile - 4, tile - 4))
        cv2.putText(t, f"true {CLASS_NAMES[y[i]]} c{F[i][0]:.1f} r{F[i][1]:.1f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
        sheet[r * tile + 2:(r + 1) * tile - 2, c * tile + 2:(c + 1) * tile - 2] = t
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]); print(f"{len(bad)} mistakes on {split} -> {out}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--show-mistakes", action="store_true"); args = ap.parse_args()
    print("weights:", WEIGHTS, "bias:", BIAS)
    Ftr, ytr = features_for_split("train"); Fva, yva = features_for_split("val")
    ptr = report("train", Ftr, ytr); pva = report("val", Fva, yva)
    # sanity: the exported module must agree with the numpy arithmetic above
    with torch.no_grad():
        from dataset import XODataset
        ds = XODataset("val"); X = torch.stack([ds[i][0] for i in range(len(ds))])
        pm = make_model()(X).argmax(1).numpy()
    print("ManualPerceptron module agrees with numpy on val:", bool((pm == pva).all()))
    figure(Ftr, ytr, Fva, yva)
    if args.show_mistakes:
        mistakes_sheet("train", Ftr, ytr, ptr, Path("data/perceptron_mistakes_train.jpg"))
        mistakes_sheet("val", Fva, yva, pva, Path("data/perceptron_mistakes_val.jpg"))


if __name__ == "__main__":
    main()
