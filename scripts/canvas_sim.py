"""
canvas_sim.py -- imitate the website's Draw tab, so we can test models on drawings at scale.

The site's canvas is 512x512 white, pen colour #111, line width 22, round caps, antialiased,
and the WHOLE canvas is sent to the preprocessing (no crop).  We draw wobbly hand-made O's and
X's with those settings at a chosen size and position.

    python scripts/canvas_sim.py          -> accuracy of every exported model by drawing size
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from contract import bgr_to_contract   # noqa: E402

PEN, INK, SIZE = 22, (17, 17, 17), 512


def _wobble(rng, pts, amount):
    return pts + rng.normal(0, amount, pts.shape)


def draw(kind: str, radius: float, rng, centre=None) -> np.ndarray:
    """One drawing: kind 'O' or 'X', radius = half the symbol's width in canvas pixels."""
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)
    if centre is None:
        m = radius + PEN
        centre = rng.uniform(m, SIZE - m, 2) if SIZE - 2 * m > 0 else np.array([SIZE / 2, SIZE / 2])
    cx, cy = centre
    if kind == "O":
        start = rng.uniform(0, 2 * np.pi); sweep = rng.uniform(1.75, 2.1) * np.pi       # gap or overlap, like a real O
        t = np.linspace(start, start + sweep, 80)
        aspect = rng.uniform(0.6, 1.0); tilt = rng.uniform(0, np.pi)
        x, y = radius * np.cos(t), radius * aspect * np.sin(t)
        pts = np.stack([cx + x * np.cos(tilt) - y * np.sin(tilt), cy + x * np.sin(tilt) + y * np.cos(tilt)], 1)
        strokes = [_wobble(rng, pts, radius * 0.02)]
    else:
        strokes = []
        for a in (np.pi / 4, 3 * np.pi / 4):
            a += rng.normal(0, 0.2); r1, r2 = radius * rng.uniform(0.8, 1.1), radius * rng.uniform(0.8, 1.1)
            p = np.array([[cx - r1 * np.cos(a), cy - r1 * np.sin(a)], [cx + r2 * np.cos(a), cy + r2 * np.sin(a)]])
            strokes.append(_wobble(rng, np.linspace(p[0], p[1], 20), radius * 0.015))
    for s in strokes:
        cv2.polylines(img, [np.round(s).astype(np.int32)], False, INK, PEN, cv2.LINE_AA)
    return img


def batch(n_each: int, radius, seed=0):
    rng = np.random.default_rng(seed)
    imgs, ys = [], []
    for kind, y in (("O", 0), ("X", 1)):
        for _ in range(n_each):
            r = rng.uniform(*radius) if isinstance(radius, tuple) else radius
            imgs.append(bgr_to_contract(draw(kind, r, rng))); ys.append(y)
    return np.stack(imgs), np.array(ys)


if __name__ == "__main__":
    import onnxruntime as ort
    names = sys.argv[1:] or ["perceptron", "mlp", "cnn"]
    sess = {n: ort.InferenceSession(f"artifacts/{n}.onnx", providers=["CPUExecutionProvider"]) for n in names}
    print("simulated website drawings, 100 O + 100 X per row; symbol width as a share of the 512 canvas")
    print(f"{'width':>6s} | " + " | ".join(f"{n:>10s} acc  O ok" for n in names))
    for frac in (0.85, 0.65, 0.5, 0.35, 0.25):
        X, y = batch(100, frac * SIZE / 2, seed=int(frac * 100))
        cells = []
        for n, s in sess.items():
            p = np.array([s.run(["logits"], {"image": x[None]})[0][0].argmax() for x in X])
            cells.append(f"{(p == y).mean():>10.0%} {(p == y)[y == 0].mean():>5.0%}")
        print(f"{frac:>6.0%} | " + " | ".join(cells))
