"""
perceptron_manual.py -- Classifier 1: ONE perceptron with weights chosen BY HAND.

    score = WEIGHTS . features + BIAS          predict X if score > 0, else O

No training, no gradient descent: edit WEIGHTS / BIAS below, re-run, read the accuracy.
The features are ink-density ratios from features.py (1.0 = average density).

    python scripts/perceptron_manual.py                 train + val accuracy, mistakes, figure
    python scripts/perceptron_manual.py --show-mistakes  also write data/perceptron_mistakes.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_NAMES, MASKS, features_for_split, extract  # noqa: E402

# ----------------------------------------------------------------------------- #
# HAND-CHOSEN WEIGHTS.  Positive weight = "more of this pushes towards X".
# BIAS is the threshold moved to the left-hand side: score > 0 means X.
# ----------------------------------------------------------------------------- #
WEIGHTS = {
    "centre":  1.0,     # X crosses in the middle -> high centre density
    "ring":   -1.0,     # O's stroke lives in the ring -> high ring density means O
    "diag":    0.0,
    "corners": 0.0,     # measured: identical for X and O (padding), so it can't help
}
BIAS = -0.3


def predict(F: np.ndarray) -> np.ndarray:
    w = np.array([WEIGHTS[k] for k in FEATURE_NAMES], dtype=np.float32)
    return (F @ w + BIAS > 0).astype(int)                 # 1 = X, 0 = O


def report(name, F, y):
    p = predict(F)
    acc = (p == y).mean()
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    majority = max(y.mean(), 1 - y.mean())
    print(f"{name:5s} acc={acc:.3f}  (always-guess-majority would be {majority:.3f}, coin flip 0.5)")
    print(f"      X right {tp:3d}   X called O {fn:3d}   O right {tn:3d}   O called X {fp:3d}")
    return p


def figure(Ftr, ytr, Fva, yva, out=Path("data/perceptron_features.png")):
    """Left: the 4 regions drawn over an X and an O.  Middle: per-feature histograms by class.
    Right: centre vs ring scatter with the hand-chosen decision line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dataset import XODataset
    ds = XODataset("train", size=32)
    ix = int(np.where(ds.labels.numpy() == 1)[0][0]); io = int(np.where(ds.labels.numpy() == 0)[0][0])
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for row, (i, nm) in enumerate([(ix, "X"), (io, "O")]):
        img = ds[i][0][0].numpy()
        for col, k in enumerate(FEATURE_NAMES):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.imshow(np.ma.masked_where(~MASKS[k], MASKS[k]), cmap="autumn", alpha=0.35, vmin=0, vmax=1)
            ax.set_title(f"{nm}: {k} = {extract(img)[col]:.2f}"); ax.axis("off")
    fig.suptitle("Feature regions (orange) over one X and one O; value = ink density in region / average density")
    fig.tight_layout(); fig.savefig(out.with_name("feature_masks.png"), dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for col, k in enumerate(FEATURE_NAMES):
        ax = axes[col]
        ax.hist(Ftr[ytr == 1, col], bins=30, alpha=0.6, label="X", color="tab:green")
        ax.hist(Ftr[ytr == 0, col], bins=30, alpha=0.6, label="O", color="tab:blue")
        ax.set_title(f"{k} (train)"); ax.legend()
    ax = axes[4]
    for F, y, mk, lab in [(Ftr, ytr, "o", "train"), (Fva, yva, "^", "val")]:
        ax.scatter(F[y == 1, 0], F[y == 1, 1], c="tab:green", marker=mk, s=14, label=f"X {lab}")
        ax.scatter(F[y == 0, 0], F[y == 0, 1], c="tab:blue", marker=mk, s=14, label=f"O {lab}")
    wc, wr = WEIGHTS["centre"], WEIGHTS["ring"]
    xs = np.linspace(0, Ftr[:, 0].max(), 50)
    if wr != 0:
        ax.plot(xs, -(wc * xs + BIAS) / wr, "k--", label="score = 0")
    ax.set_xlabel("centre"); ax.set_ylabel("ring"); ax.set_title("decision line (centre & ring weights only)")
    ax.set_ylim(0, 3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print("figures ->", out.with_name("feature_masks.png"), out)


def mistakes_sheet(split, F, y, p, out):
    """Tile the misclassified crops so you can see WHY the line missed them."""
    from dataset import XODataset, CLASS_NAMES
    ds = XODataset(split, size=32)
    bad = np.where(p != y)[0]
    if len(bad) == 0:
        print("no mistakes on", split); return
    tile, cols = 96, 10
    rows = int(np.ceil(len(bad) / cols))
    sheet = np.full((rows * tile, cols * tile, 3), 255, np.uint8)
    for n, i in enumerate(bad):
        r, c = divmod(n, cols)
        t = cv2.resize(ds.raw[i], (tile - 4, tile - 4))
        f = F[i]
        cv2.putText(t, f"true {CLASS_NAMES[y[i]]} c{f[0]:.1f} r{f[1]:.1f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
        sheet[r * tile + 2:(r + 1) * tile - 2, c * tile + 2:(c + 1) * tile - 2] = t
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(bad)} mistakes on {split} ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-mistakes", action="store_true")
    ap.add_argument("--test", action="store_true", help="ALSO score the held-out test photo (do this once, at the end)")
    args = ap.parse_args()
    print("weights:", WEIGHTS, "bias:", BIAS)
    Ftr, ytr = features_for_split("train")
    Fva, yva = features_for_split("val")
    ptr = report("train", Ftr, ytr)
    pva = report("val", Fva, yva)
    figure(Ftr, ytr, Fva, yva)
    if args.show_mistakes:
        mistakes_sheet("train", Ftr, ytr, ptr, Path("data/perceptron_mistakes_train.jpg"))
        mistakes_sheet("val", Fva, yva, pva, Path("data/perceptron_mistakes_val.jpg"))
    if args.test:
        Fte, yte = features_for_split("test")
        report("test", Fte, yte)


if __name__ == "__main__":
    main()
