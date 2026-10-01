"""
export_playground.py -- export all three classifiers to the class website's format and
prove, end to end, that the exported files behave like the PyTorch models on real crops.

    python scripts/export_playground.py              -> artifacts/{perceptron,mlp,cnn}.onnx

Two checks happen:
  1. the playground's own exporter compares PyTorch vs ONNX on 8 synthetic inputs + 16 of
     our real crops (tolerance 1e-5), and enforces the [1,1,64,64] -> [1,2] contract;
  2. we then run the SAME path the browser will: PNG file -> playground preprocessing (PIL,
     EXIF-aware) -> onnxruntime -> argmax, over every val and test crop, and report accuracy.
     If this number matches compare.py, the website will see what we saw.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent))
from playground_lib.playground_export import export_for_playground   # noqa: E402
from contract import file_to_contract                                 # noqa: E402
from dataset import load_rows, LABELS, MANIFEST                       # noqa: E402
import compare                                                        # noqa: E402

ART = Path("artifacts")


def browser_path_accuracy(onnx_path: Path, split: str):
    """Exactly what the website does, minus the browser: file -> contract tensor -> ONNX -> argmax."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rows = load_rows(split); correct = 0
    for r in rows:
        x = file_to_contract(MANIFEST.parent / r["file"])[None]          # [1,1,64,64] float32
        logits = sess.run(["logits"], {"image": x})[0]
        correct += int(logits.argmax() == LABELS[r["label"]])
    return correct / len(rows), len(rows)


def main():
    ART.mkdir(exist_ok=True)
    models, _, _ = compare.all_models()
    # a handful of real preprocessed crops for the exporter's extra parity check
    rows = load_rows("val")[:16]
    samples = [torch.from_numpy(file_to_contract(MANIFEST.parent / r["file"])[None]) for r in rows]
    print(f"{'model':11s} {'file':28s} {'size':>8s}   {'val acc':>8s} {'test acc':>9s}   (file -> contract -> ONNX, like the website)")
    for name, m in models.items():
        path = export_for_playground(m, ART / f"{name}.onnx", class_order=("O", "X"), sample_inputs=samples)
        va, nv = browser_path_accuracy(path, "val"); te, nt = browser_path_accuracy(path, "test")
        print(f"{name:11s} {str(path):28s} {path.stat().st_size / 1024:7.0f}K   {va:8.1%} {te:9.1%}")
    print("\nUpload any of these .onnx files on the class website; they declare contract ox-gray64-v1, classes O,X.")


if __name__ == "__main__":
    main()
