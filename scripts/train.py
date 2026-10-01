"""
train.py -- Step 4/5: train a classifier and evaluate it on the validation photos.

    python scripts/train.py --model mlp                      default settings
    python scripts/train.py --model mlp --epochs 80 --lr 5e-4 --hidden 256 64
    python scripts/train.py --model mlp --binarize            ablation: 0/1 pixels
    python scripts/train.py --model mlp --no-augment          ablation: see it overfit

Writes runs/<name>/  best.pt (weights at best val accuracy), history.json,
curves.png (loss + accuracy per epoch), confusion.png, mistakes_val.jpg.

THE LOOP, in words:
    for each epoch:
        for each mini-batch of training images:
            forward   : logits = model(images)
            loss      : how wrong, as one number (binary cross-entropy)
            backward  : loss.backward() computes d(loss)/d(weight) for every weight
            step      : optimizer nudges every weight a little downhill
        evaluate on val (no augmentation, no dropout, no gradients)
        if val accuracy is the best so far: save the weights   (= early stopping)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import XODataset, CLASS_NAMES  # noqa: E402
from models import build, count_params      # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"         # train / val, fixed everywhere


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """Average loss and accuracy over a loader.  eval() turns dropout off; no_grad skips
    gradient bookkeeping.  Returns also every prediction so we can draw a confusion matrix."""
    model.eval()
    tot_loss, correct, n, preds, labels = 0.0, 0, 0, [], []
    for x, y in loader:
        logits = model(x)
        tot_loss += loss_fn(logits, y.float()).item() * len(y)
        p = (logits > 0).long()
        correct += (p == y).sum().item(); n += len(y)
        preds.append(p); labels.append(y)
    return tot_loss / n, correct / n, torch.cat(preds).numpy(), torch.cat(labels).numpy()


def plot_curves(hist, out: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ylabel in [(a1, "loss", "binary cross-entropy"), (a2, "acc", "accuracy")]:
        ax.plot(ep, hist[f"train_{key}"], color=BLUE, lw=2, label="train")
        ax.plot(ep, hist[f"val_{key}"], color=ORANGE, lw=2, label="val")
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(frameon=False)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    a2.set_ylim(0.4, 1.02)
    best = int(np.argmax(hist["val_acc"]))
    a2.annotate(f"best val {hist['val_acc'][best]:.3f} @ epoch {best + 1}", (best + 1, hist["val_acc"][best]),
                xytext=(10, -30), textcoords="offset points", fontsize=9, arrowprops=dict(arrowstyle="-", lw=1))
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_confusion(preds, labels, out: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.zeros((2, 2), int)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return cm


def mistakes_sheet(ds, preds, labels, out: Path):
    bad = np.where(preds != labels)[0]
    if len(bad) == 0:
        return 0
    tile, cols = 96, 10
    sheet = np.full((int(np.ceil(len(bad) / cols)) * tile, cols * tile, 3), 255, np.uint8)
    for n, i in enumerate(bad):
        r, c = divmod(n, cols)
        t = cv2.resize(ds.raw[i], (tile - 4, tile - 4))
        cv2.putText(t, f"true {CLASS_NAMES[labels[i]]} -> {CLASS_NAMES[preds[i]]}", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
        sheet[r * tile + 2:(r + 1) * tile - 2, c * tile + 2:(c + 1) * tile - 2] = t
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return len(bad)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    ap.add_argument("--name", default=None, help="run folder name (default: model name + flags)")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64], help="MLP hidden layer sizes")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay (L2 penalty on weights)")
    ap.add_argument("--binarize", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    name = args.name or f"{args.model}" + ("_bin" if args.binarize else "") + ("_noaug" if args.no_augment else "")
    run = Path("runs") / name
    run.mkdir(parents=True, exist_ok=True)

    train_ds = XODataset("train", size=args.size, binarize=args.binarize, augment=not args.no_augment)
    val_ds = XODataset("val", size=args.size, binarize=args.binarize, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256)

    kw = dict(hidden=tuple(args.hidden), dropout=args.dropout) if args.model == "mlp" else dict(dropout=args.dropout)
    model = build(args.model, args.size, **kw)
    print(f"{args.model}: {count_params(model):,} parameters, {len(train_ds)} train / {len(val_ds)} val images")

    # Binary cross-entropy on the raw logit (numerically safer than sigmoid + BCELoss).
    loss_fn = nn.BCEWithLogitsLoss()
    # Adam: gradient descent with a per-weight adaptive step size.  weight_decay shrinks
    # weights every step (L2 regularisation): big weights = confident, brittle model.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()                                   # dropout ON
        tot_loss, correct, n = 0.0, 0, 0
        for x, y in train_dl:
            logits = model(x)                           # forward
            loss = loss_fn(logits, y.float())           # how wrong (one number)
            opt.zero_grad()                             # clear last step's gradients
            loss.backward()                             # d loss / d every weight
            opt.step()                                  # move every weight a little downhill
            tot_loss += loss.item() * len(y); correct += ((logits > 0).long() == y).sum().item(); n += len(y)
        tr_loss, tr_acc = tot_loss / n, correct / n
        va_loss, va_acc, preds, labels = evaluate(model, val_dl, loss_fn)
        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc)):
            hist[k].append(v)
        flag = ""
        if va_acc > best_acc:                           # early stopping: keep the best epoch's weights
            best_acc = va_acc
            torch.save(model.state_dict(), run / "best.pt")
            flag = " *"
        if epoch % 5 == 0 or epoch == 1 or flag:
            print(f"epoch {epoch:3d}  train loss {tr_loss:.3f} acc {tr_acc:.3f}   val loss {va_loss:.3f} acc {va_acc:.3f}{flag}")

    json.dump(dict(args=vars(args), history=hist, best_val_acc=best_acc), open(run / "history.json", "w"), indent=1)
    plot_curves(hist, run / "curves.png", f"{name}: {count_params(model):,} params")
    model.load_state_dict(torch.load(run / "best.pt"))
    _, va_acc, preds, labels = evaluate(model, val_dl, loss_fn)
    cm = plot_confusion(preds, labels, run / "confusion.png", f"{name} val acc {va_acc:.3f}")
    n_bad = mistakes_sheet(val_ds, preds, labels, run / "mistakes_val.jpg")
    print(f"\nbest val acc {va_acc:.3f}   confusion [[O->O {cm[0,0]}, O->X {cm[0,1]}], [X->O {cm[1,0]}, X->X {cm[1,1]}]]")
    print(f"outputs -> {run}/  (curves.png, confusion.png, mistakes_val.jpg with {n_bad} mistakes)")


if __name__ == "__main__":
    main()
