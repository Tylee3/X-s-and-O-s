"""
compare_runs.py -- old vs new models on everything we have, to decide which to ship.

    python scripts/compare_runs.py mlp mlp_cs cnn cnn_cs

Columns: train / val / test photo crops, the 104 hand-boxed live crops, simulated website
drawings at 85/65/50/35/25% of the canvas, and those drawings' O accuracy at 35%.
"""
import csv, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from compare import load_trained                     # noqa: E402
from contract import file_to_contract                # noqa: E402
from dataset import load_rows, MANIFEST, LABELS      # noqa: E402
from canvas_sim import batch                         # noqa: E402


def sets():
    def data(rows, root):
        return torch.stack([torch.from_numpy(file_to_contract(root / r["file"])) for r in rows]), torch.tensor([LABELS[r["label"]] for r in rows])
    out = {s: data(load_rows(s), MANIFEST.parent) for s in ("train", "val", "test")}
    live = [r for r in csv.DictReader(open("data/live_crops/manifest.csv")) if r["label"] in ("O", "X")]
    out["live"] = data(live, Path("data/live_crops"))
    # uncropped-photo proxy: real val+test crops placed at a fraction of a 512 frame, surround = the crop's own
    # blurred background (what you get photographing ONE shape from further back without cropping)
    import cv2
    real = load_rows("val") + load_rows("test")
    crops = [cv2.imread(str(MANIFEST.parent / r["file"])) for r in real]
    yr = torch.tensor([LABELS[r["label"]] for r in real])
    from contract import bgr_to_contract
    for frac in (0.5, 0.35):
        Xs = []
        for c in crops:
            w = int(frac * 512); bg = cv2.resize(cv2.resize(cv2.medianBlur(c, 31), (w, w)), (512, 512)); o = (512 - w) // 2
            bg[o:o + w, o:o + w] = cv2.resize(c, (w, w), interpolation=cv2.INTER_AREA); Xs.append(bgr_to_contract(bg))
        out[f"photo{int(frac * 100)}"] = (torch.from_numpy(np.stack(Xs)), yr)
    for frac in (0.85, 0.65, 0.5, 0.35, 0.25):
        X, y = batch(100, frac * 256, seed=1000 + int(frac * 100))       # different seed from canvas_sim's table
        out[f"draw{int(frac * 100)}"] = (torch.from_numpy(X), torch.from_numpy(y))
    return out


@torch.no_grad()
def main(runs):
    S = sets()
    print(f"{'run':10s} " + " ".join(f"{k:>7s}" for k in S) + "   O@draw35")
    for run in runs:
        m, _ = load_trained(run)
        cells, o35 = [], None
        for k, (X, y) in S.items():
            p = m(X).argmax(1); cells.append(f"{float((p == y).float().mean()):7.1%}")
            if k == "draw35": o35 = float((p == y)[y == 0].float().mean())
        print(f"{run:10s} " + " ".join(cells) + f"   {o35:7.1%}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["mlp", "mlp_cs", "cnn"])
