"""
compare.py -- Step 6: score all three classifiers on the same data and make the figures.

    python scripts/compare.py            train + val only
    python scripts/compare.py --test     ALSO the held-out test photo (photo11). Do this once.

Writes results/:
    results.md            accuracy table (markdown, paste into the write-up)
    accuracy.png          bar chart, one bar group per classifier
    confusion_<split>.png the three confusion matrices side by side
    curves_mlp_vs_cnn.png training curves of the two trained models on one pair of axes
    mistakes_<split>.jpg  every test/val crop any classifier got wrong, with each model's answer
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import XODataset, CLASS_NAMES                 # noqa: E402
from features import extract, FEATURE_NAMES                # noqa: E402
import perceptron_manual as P                              # noqa: E402  (WEIGHTS / BIAS live there)
from models import build                                   # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
OUT = Path("results")


def load_trained(name, size=32):
    hist = json.load(open(Path("runs") / name / "history.json"))
    a = hist["args"]
    kw = dict(hidden=tuple(a["hidden"]), dropout=a["dropout"]) if a["model"] == "mlp" else dict(dropout=a["dropout"])
    m = build(a["model"], size, **kw)
    m.load_state_dict(torch.load(Path("runs") / name / "best.pt"))
    m.eval()
    return m, hist


@torch.no_grad()
def predictions(split, mlp, cnn):
    ds = XODataset(split, size=32)
    X = torch.stack([ds[i][0] for i in range(len(ds))])          # [N,1,32,32]
    y = ds.labels.numpy()
    feats = np.stack([extract(x[0].numpy()) for x in X])
    preds = {
        "perceptron": P.predict(feats),
        "mlp": (mlp(X) > 0).long().numpy(),
        "cnn": (cnn(X) > 0).long().numpy(),
    }
    return ds, y, preds


def confusion(p, y):
    cm = np.zeros((2, 2), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def plot_accuracy(table, splits):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    models = ["perceptron", "mlp", "cnn"]
    colors = {"train": BLUE, "val": ORANGE, "test": AQUA}
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(splits)
    for k, s in enumerate(splits):
        xs = np.arange(len(models)) + (k - (len(splits) - 1) / 2) * w
        vals = [table[m][s] for m in models]
        ax.bar(xs, vals, width=w * 0.92, color=colors[s], label=s)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.005, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="#888", lw=1, ls=":"); ax.text(len(models) - 0.55, 0.505, "coin flip", fontsize=8, color="#666")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(["manual perceptron", "MLP", "CNN"])
    ax.set_ylim(0.4, 1.08); ax.set_ylabel("accuracy"); ax.legend(frameon=False, ncol=len(splits), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "accuracy.png", dpi=120); plt.close(fig)


def plot_confusions(cms, split):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (name, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{name} ({split}) acc {np.trace(cm) / cm.sum():.1%}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"confusion_{split}.png", dpi=120); plt.close(fig)


def plot_curves(h_mlp, h_cnn):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for h, name, col in [(h_mlp["history"], "MLP", BLUE), (h_cnn["history"], "CNN", ORANGE)]:
        ep = range(1, len(h["val_loss"]) + 1)
        a1.plot(ep, h["train_loss"], color=col, lw=1.5, ls=":", label=f"{name} train")
        a1.plot(ep, h["val_loss"], color=col, lw=2, label=f"{name} val")
        a2.plot(ep, h["train_acc"], color=col, lw=1.5, ls=":", label=f"{name} train")
        a2.plot(ep, h["val_acc"], color=col, lw=2, label=f"{name} val")
    a1.set_ylabel("binary cross-entropy"); a2.set_ylabel("accuracy"); a2.set_ylim(0.6, 1.02)
    for ax in (a1, a2):
        ax.set_xlabel("epoch"); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.suptitle("MLP vs CNN, same data, loss, optimizer (dotted = train, solid = val)")
    fig.tight_layout(); fig.savefig(OUT / "curves_mlp_vs_cnn.png", dpi=120); plt.close(fig)


def mistakes_sheet(ds, y, preds, split):
    any_bad = np.where(np.any(np.stack([p != y for p in preds.values()]), axis=0))[0]
    if len(any_bad) == 0:
        return 0
    tile, cols = 110, 8
    sheet = np.full((int(np.ceil(len(any_bad) / cols)) * tile, cols * tile, 3), 255, np.uint8)
    for n, i in enumerate(any_bad):
        r, c = divmod(n, cols)
        t = cv2.resize(ds.raw[i], (tile - 4, tile - 4))
        cv2.putText(t, f"true {CLASS_NAMES[y[i]]}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        line = " ".join(f"{k[0].upper()}:{CLASS_NAMES[p[i]]}" for k, p in preds.items())
        cv2.putText(t, line, (2, tile - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 200), 1, cv2.LINE_AA)
        sheet[r * tile + 2:(r + 1) * tile - 2, c * tile + 2:(c + 1) * tile - 2] = t
    cv2.imwrite(str(OUT / f"mistakes_{split}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return len(any_bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="also score photo11, the held-out test. Once.")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    mlp, h_mlp = load_trained("mlp")
    cnn, h_cnn = load_trained("cnn")
    splits = ["train", "val"] + (["test"] if args.test else [])

    table = {m: {} for m in ("perceptron", "mlp", "cnn")}
    sizes = {}
    for split in splits:
        ds, y, preds = predictions(split, mlp, cnn)
        sizes[split] = len(y)
        cms = {}
        for name, p in preds.items():
            table[name][split] = float((p == y).mean())
            cms[name] = confusion(p, y)
        plot_confusions(cms, split)
        n_bad = mistakes_sheet(ds, y, preds, split)
        print(f"{split}: {len(y)} images, {n_bad} got wrong by at least one classifier")

    plot_accuracy(table, splits)
    plot_curves(h_mlp, h_cnn)

    lines = ["| classifier | params | " + " | ".join(f"{s} (n={sizes[s]})" for s in splits) + " |",
             "|---|---|" + "---|" * len(splits)]
    params = {"perceptron": f"{sum(1 for v in P.WEIGHTS.values() if v) } weights + bias (hand-set)",
              "mlp": f"{sum(p.numel() for p in mlp.parameters()):,}", "cnn": f"{sum(p.numel() for p in cnn.parameters()):,}"}
    for m in table:
        lines.append(f"| {m} | {params[m]} | " + " | ".join(f"{table[m][s]:.1%}" for s in splits) + " |")
    md = "\n".join(lines)
    (OUT / "results.md").write_text(md + "\n")
    print("\n" + md)
    print(f"\nfigures -> {OUT}/")


if __name__ == "__main__":
    main()
