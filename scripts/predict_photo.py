"""
predict_photo.py -- run the exported models over every crop of a segmented photo and draw the
predictions on the photo.  This is the whole system end to end on a NEW photo:
segment.py boxes -> crop files -> playground preprocessing -> ONNX model -> label.

    python scripts/predict_photo.py data/live_crops            (all photos in that crops folder)

Writes <crops>/<photo>/<photo>_pred.jpg (green X / blue O with the CNN's confidence; small text
underneath gives the MLP and perceptron answers) and prints how often the three models agree.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from contract import file_to_contract   # noqa: E402

CLASSES = ["O", "X"]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/live_crops")
    sess = {n: ort.InferenceSession(f"artifacts/{n}.onnx", providers=["CPUExecutionProvider"]) for n in ("cnn", "mlp", "perceptron")}
    rows = list(csv.DictReader(open(root / "manifest.csv")))
    for photo in sorted({r["photo"] for r in rows}):
        img_path = next(Path("data/live").glob(photo + ".*"), None) or next(Path("data/raw").glob(photo + ".*"))
        img = cv2.imread(str(img_path))
        agree = Counter()
        for r in (r for r in rows if r["photo"] == photo):
            x = file_to_contract(root / r["file"])[None]
            out = {}
            for n, s in sess.items():
                l = s.run(["logits"], {"image": x})[0][0]; p = np.exp(l - l.max()); p /= p.sum()
                out[n] = (CLASSES[int(p.argmax())], float(p.max()))
            agree["all three agree" if len({v[0] for v in out.values()}) == 1 else "disagree"] += 1
            lab, conf = out["cnn"]
            color = (0, 200, 0) if lab == "X" else (255, 120, 0)
            x0, y0, w, h = (int(r[k]) for k in ("x", "y", "w", "h"))
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), color, 3)
            cv2.putText(img, f"{lab} {conf:.0%}", (x0, max(22, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            cv2.putText(img, f"M:{out['mlp'][0]} P:{out['perceptron'][0]}", (x0, y0 + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
        out_path = root / photo / f"{photo}_pred.jpg"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        n = sum(agree.values())
        print(f"{photo}: {n} crops, {agree['all three agree']} where all three models agree -> {out_path}")


if __name__ == "__main__":
    main()
