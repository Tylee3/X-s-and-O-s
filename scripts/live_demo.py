"""
live_demo.py -- draw X's and O's on paper/whiteboard, hold it to the webcam, watch the
model call them live.  This is the closest thing to the in-class test you can rehearse.

    python scripts/live_demo.py                     CNN, webcam 0
    python scripts/live_demo.py --model mlp
    python scripts/live_demo.py --camera 1           if the built-in cam isn't index 0

Keys while it's running:
    m   cycle CNN -> MLP -> manual perceptron
    [ ] lower / raise the ink-detection threshold (same idea as segment.py --thresh)
    q   quit

WHAT'S HAPPENING EACH FRAME (reuses segment.py's ink detection, simplified: no "find the
board" step, because here you're holding the paper close and filling most of the frame):
    1. ink = darkest of R,G,B at every pixel            (segment.py's ink_channel)
    2. darkness vs local background via a morphological closing (segment.py's black top-hat)
    3. threshold -> ink mask, small dilation -> connected components -> one box per shape
    4. for each box: crop it, convert it EXACTLY as the class website would (contract.py),
       feed the model, draw the box in green (X) or blue (O) with the confidence
No training happens here -- this is pure inference, same forward pass as evaluate().
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import segment  # noqa: E402  (we poke segment.INK_THRESH directly, like segment.py's own --thresh flag)
from segment import ink_channel, darkness_vs_background, ink_mask, find_shapes  # noqa: E402
from contract import bgr_to_contract                                           # noqa: E402
from dataset import CLASS_NAMES                                                # noqa: E402
import compare                                                                 # noqa: E402

# live video is smaller and cleaner than a photo, so these differ from segment.py's defaults
MIN_SIDE_PX = 40          # ignore boxes smaller than this (noise, finger, marker cap)
MAX_SIDE_FRAC = 0.9       # ignore boxes basically the whole frame (nothing drawn yet)
PAD_FRAC = 0.25
WORK_LONG_SIDE = 640      # detection runs on a downscaled copy for speed; crops come from the full frame


def detect_boxes(frame_bgr: np.ndarray, thresh: int):
    """Same pipeline as segment.py, minus the 'find the board' step. Returns boxes in
    FULL-FRAME pixel coordinates even though detection itself runs on a smaller copy."""
    H, W = frame_bgr.shape[:2]
    scale = WORK_LONG_SIDE / max(H, W) if max(H, W) > WORK_LONG_SIDE else 1.0
    work = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr
    ink = ink_channel(work)
    dark = darkness_vs_background(ink, max(work.shape[:2]))
    segment.INK_THRESH = thresh                            # same knob as segment.py's --thresh
    mask = ink_mask(dark)
    shapes = find_shapes(mask, max(work.shape[:2]))
    boxes = []
    min_side, max_side = MIN_SIDE_PX * scale, MAX_SIDE_FRAC * max(work.shape[:2])
    for s in shapes:
        side = max(s["w"], s["h"])
        if side < min_side or side > max_side:
            continue
        boxes.append(dict(x=s["x"] / scale, y=s["y"] / scale, w=s["w"] / scale, h=s["h"] / scale))
    return boxes


def square_crop_live(frame, box, pad=PAD_FRAC):
    H, W = frame.shape[:2]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * pad)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return frame[y0:y1, x0:x1]


@torch.no_grad()
def classify(model, crop_bgr):
    """Same path as the website: crop -> playground contract tensor -> model -> softmax over (O, X)."""
    x = torch.from_numpy(bgr_to_contract(crop_bgr)).unsqueeze(0)              # [1,1,64,64] in [-1,1]
    probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx])


BLACK_FRAME_STD = 2.0    # a real image's pixel std is >>2 even in a dim room; an all-zero frame is ~0


def open_camera(index: int):
    """Open the webcam and don't hand it back until it's actually delivering a real picture.

    Why this is needed: `VideoCapture.isOpened()` can return True the instant the OS hands
    over a camera handle, before the sensor has actually started streaming. Worse, on macOS
    when the camera permission is incomplete, `.read()` often reports SUCCESS (ok=True) and
    hands back a frame that is just all-zero pixels, instead of failing outright -- the OS
    lets the connection open but silently withholds real video. So checking `ok` is not
    enough: we also check that the frame has real contrast (std well above 0) before trusting it.
    """
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == "darwin" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        saw_black = False
        for _ in range(90):                                   # ~3s at 30fps-worth of attempts
            ok, frame = cap.read()
            if ok and frame is not None:
                if frame.std() > BLACK_FRAME_STD:
                    return cap                                 # a real picture, not just a successful handshake
                saw_black = True
            cv2.waitKey(33)
        cap.release()
    extra = (
        "\nThe camera opened and reported frames successfully, but every one was solid black.\n"
        "That specific symptom (not a hard failure) almost always means macOS allowed the\n"
        "CONNECTION but is still withholding actual video -- check for the small green/orange\n"
        "camera-in-use dot in the menu bar: if it's NOT lit, permission isn't really active yet.\n"
    ) if saw_black else ""
    raise SystemExit(
        f"could not get a real picture from camera {index} after multiple attempts.{extra}\n"
        f"Most likely causes:\n"
        f"  1. macOS hasn't granted THIS SPECIFIC app camera access: System Settings ->\n"
        f"     Privacy & Security -> Camera -> find your actual terminal app (Terminal.app and\n"
        f"     iTerm are listed separately; VS Code's integrated terminal is its own entry too)\n"
        f"     and enable it, then QUIT AND FULLY REOPEN that app -- a permission grant does not\n"
        f"     apply to an already-running process.\n"
        f"  2. Another app (Zoom, Photo Booth, a browser tab) is holding the camera open --\n"
        f"     close it and try again.\n"
        f"  3. Wrong index: try --camera 1 (or 2) if this Mac has more than one camera.\n"
        f"  4. A physical lens cover/privacy shutter is closed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["cnn", "mlp", "perceptron"], default="cnn")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--thresh", type=int, default=35)
    args = ap.parse_args()

    models, _, _ = compare.all_models()                 # perceptron, mlp, cnn (same weights the .onnx files hold)
    order = ["cnn", "mlp", "perceptron"]
    current = args.model
    thresh = args.thresh

    cap = open_camera(args.camera)
    win = "live X/O demo  (m=model [/]=threshold q=quit)"
    cv2.namedWindow(win)
    fps_t, fps = time.time(), 0.0
    misses = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            # A single bad grab is normal (dropped USB frame, momentary focus hunt) and
            # should NOT end the session -- only give up after many in a row, which means
            # the camera actually went away (unplugged, put to sleep, app lost access).
            misses += 1
            if misses > 60:
                print("camera stopped responding (60 failed grabs in a row) -- is another app using it, "
                      "or did the laptop lid close?")
                break
            cv2.waitKey(10)
            continue
        misses = 0
        frame = cv2.flip(frame, 1)                                     # mirror: natural when holding paper up

        if frame.std() <= BLACK_FRAME_STD:
            # A frame came back "ok" but is blank: lens cap, app lost permission mid-run, etc.
            # Show this instead of silently running detection on nothing, which would just
            # look like the model "doesn't see" anything with no explanation.
            cv2.putText(frame, "no picture from camera (lens covered, or permission dropped)",
                        (20, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            continue

        boxes = detect_boxes(frame, thresh)
        for b in boxes:
            crop = square_crop_live(frame, b)
            if crop is None:
                continue
            label, conf = classify(models[current], crop)
            color = (0, 200, 0) if label == "X" else (255, 120, 0)        # green X, blue O, same as edit_boxes.py
            x0, y0 = int(b["x"]), int(b["y"])
            x1, y1 = int(b["x"] + b["w"]), int(b["y"] + b["h"])
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, f"{label} {conf:.0%}", (x0, max(18, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        now = time.time(); fps = 0.9 * fps + 0.1 * (1 / max(1e-3, now - fps_t)); fps_t = now
        status = f"model={current.upper()}  thresh={thresh}  shapes={len(boxes)}  {fps:4.1f} fps"
        cv2.putText(frame, status, (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("m"):
            current = order[(order.index(current) + 1) % len(order)]      # cnn -> mlp -> perceptron -> cnn
        elif k == ord("["):
            thresh = max(5, thresh - 5)
        elif k == ord("]"):
            thresh = min(100, thresh + 5)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
