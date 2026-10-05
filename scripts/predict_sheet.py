"""
predict_sheet.py -- contact sheet of every crop in a crops folder with the CNN's answer, for
fast eyeballing of a new photo.  Red frame = the three models disagree (look twice).

    python scripts/predict_sheet.py data/live_crops       -> <crops>/<photo>/<photo>_pred_sheet.jpg
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from contract import file_to_contract   # noqa: E402

CLASSES = ["O", "X"]; TILE, COLS = 110, 10


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/live_crops")
    sess = {n: ort.InferenceSession(f"artifacts/{n}.onnx", providers=["CPUExecutionProvider"]) for n in ("cnn", "mlp", "perceptron")}
    rows = list(csv.DictReader(open(root / "manifest.csv")))
    for photo in sorted({r["photo"] for r in rows}):
        mine = [r for r in rows if r["photo"] == photo]
        sheet = np.full((math.ceil(len(mine) / COLS) * TILE, COLS * TILE, 3), 255, np.uint8)
        for n, r in enumerate(mine):
            x = file_to_contract(root / r["file"])[None]
            labs = {}
            for m, s in sess.items():
                l = s.run(["logits"], {"image": x})[0][0]; p = np.exp(l - l.max()); p /= p.sum(); labs[m] = (CLASSES[int(p.argmax())], float(p.max()))
            t = cv2.resize(cv2.imread(str(root / r["file"])), (TILE - 6, TILE - 6))
            lab, conf = labs["cnn"]
            color = (0, 170, 0) if lab == "X" else (220, 100, 0)
            cv2.putText(t, f"{r['idx']}: {lab} {conf:.0%}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
            frame = (0, 0, 255) if len({v[0] for v in labs.values()}) > 1 else (200, 200, 200)
            t = cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=frame)
            rr, cc = divmod(n, COLS); sheet[rr * TILE:(rr + 1) * TILE, cc * TILE:(cc + 1) * TILE] = t
        out = root / photo / f"{photo}_pred_sheet.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]); print("wrote", out)


if __name__ == "__main__":
    main()
