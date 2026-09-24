"""
edit_boxes.py -- fix segmentation mistakes by hand: delete, draw, merge and label boxes.

Usage:
    python scripts/edit_boxes.py photo1            (one photo at a time)

Mouse:
    click inside a box        select it (the smallest box under the cursor)
    shift + click             add another box to the selection (for merging)
    click + drag on empty     draw a new box
    scroll wheel              zoom in/out around the cursor
Keys:
    d          delete selected box(es)
    m          merge selected boxes into one box (their union)
    x / o / j  label selected box(es) X / O / junk
    + / -      zoom in / out       i k h l   pan up / down / left / right
    0          reset view
    u          undo last edit
    S          save: rewrite manifest rows for this photo, re-crop, redraw overlay + sheet
    q          quit (asks you to press q again if there are unsaved edits)

Box colours:  green = X,  blue = O,  grey = junk,  orange = unlabeled,  red = selected
All coordinates in the manifest are in ORIGINAL photo pixels, so edits are exact.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from segment import square_crop, contact_sheet  # noqa: E402  (reuse Step-1 code, same crops)

FIELDS = ["file", "photo", "idx", "x", "y", "w", "h", "ink_px", "aspect", "fill", "flags", "label"]
VIEW_W, VIEW_H = 1500, 880          # on-screen canvas size
COLORS = {"X": (0, 200, 0), "O": (255, 120, 0), "junk": (140, 140, 140), "": (0, 140, 255)}
SELECTED = (0, 0, 255)


# --------------------------------------------------------------------------- #
# Pure state operations (no GUI) so they can be tested without a window.
# --------------------------------------------------------------------------- #
class Boxes:
    """The editable list of boxes for one photo.  Each box is a manifest row (dict)."""

    def __init__(self, rows, photo):
        self.photo = photo
        self.rows = [dict(r) for r in rows]           # copy so undo can hold snapshots
        self.next_idx = 1 + max((int(r["idx"]) for r in self.rows), default=-1)
        self.history = []
        self.dirty = False

    def _snapshot(self):
        self.history.append([dict(r) for r in self.rows])
        self.dirty = True

    def undo(self):
        if self.history:
            self.rows = self.history.pop()
            self.dirty = True

    def delete(self, sel):
        if not sel:
            return
        self._snapshot()
        self.rows = [r for i, r in enumerate(self.rows) if i not in sel]

    def add(self, x, y, w, h):
        self._snapshot()
        r = dict(file=f"{self.photo}/{self.photo}_{self.next_idx:03d}.png", photo=self.photo, idx=self.next_idx,
                 x=int(x), y=int(y), w=int(w), h=int(h), ink_px="", aspect=round(max(w, h) / max(1, min(w, h)), 2),
                 fill="", flags="manual", label="")
        self.next_idx += 1
        self.rows.append(r)
        return len(self.rows) - 1

    def merge(self, sel):
        """Replace the selected boxes by one box covering all of them (fixes an X split into two strokes)."""
        if len(sel) < 2:
            return None
        self._snapshot()
        picked = [self.rows[i] for i in sel]
        x0 = min(int(r["x"]) for r in picked); y0 = min(int(r["y"]) for r in picked)
        x1 = max(int(r["x"]) + int(r["w"]) for r in picked); y1 = max(int(r["y"]) + int(r["h"]) for r in picked)
        labels = {r["label"] for r in picked if r["label"]}
        self.rows = [r for i, r in enumerate(self.rows) if i not in sel]
        self.history.pop(); self.dirty = True          # add() snapshots again; keep a single undo step
        new = self.add(x0, y0, x1 - x0, y1 - y0)
        if len(labels) == 1:
            self.rows[new]["label"] = labels.pop()
        return new

    def set_label(self, sel, label):
        if not sel:
            return
        self._snapshot()
        for i in sel:
            self.rows[i]["label"] = label

    def hit(self, x, y):
        """Index of the smallest box containing original-pixel point (x, y), or None."""
        best, best_area = None, None
        for i, r in enumerate(self.rows):
            bx, by, bw, bh = (int(r[k]) for k in ("x", "y", "w", "h"))
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if best_area is None or bw * bh < best_area:
                    best, best_area = i, bw * bh
        return best


def load_manifest(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_all(manifest: Path, other_rows, boxes: Boxes, orig: np.ndarray, out_root: Path):
    """Write the manifest, delete crops of removed boxes, cut crops for new ones, redraw overlay + sheet."""
    photo_dir = out_root / boxes.photo
    keep = {r["file"] for r in boxes.rows}
    for png in photo_dir.glob(f"{boxes.photo}_[0-9][0-9][0-9].png"):
        if str(png.relative_to(out_root)) not in keep:
            png.unlink()                                                # crop of a deleted / merged box
    crops, tags = [], []
    for r in boxes.rows:
        p = out_root / r["file"]
        if not p.exists():
            box = {k: int(r[k]) for k in ("x", "y", "w", "h")}
            cv2.imwrite(str(p), square_crop(orig, box, 1.0))            # scale 1.0: already original pixels
        crops.append(cv2.imread(str(p)))
        tags.append(f"{r['idx']}{r['label'][:1] if r['label'] else '?'}")
    boxes.rows.sort(key=lambda r: int(r["idx"]))
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(other_rows + boxes.rows)
    # overlay at working resolution, like segment.py
    scale = min(1.0, 2400 / max(orig.shape[:2]))
    ov = cv2.resize(orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for r in boxes.rows:
        x, y, w_, h_ = (int(int(r[k]) * scale) for k in ("x", "y", "w", "h"))
        c = COLORS[r["label"]]
        cv2.rectangle(ov, (x, y), (x + w_, y + h_), c, 2)
        cv2.putText(ov, f"{r['idx']}{r['label'][:1]}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cv2.imwrite(str(photo_dir / f"{boxes.photo}_overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(photo_dir / f"{boxes.photo}_sheet.jpg"), contact_sheet(crops, tags), [cv2.IMWRITE_JPEG_QUALITY, 90])
    boxes.dirty = False
    boxes.history.clear()


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class Editor:
    def __init__(self, orig, boxes: Boxes):
        self.orig, self.boxes = orig, boxes
        H, W = orig.shape[:2]
        self.base = min(VIEW_W / W, VIEW_H / H)      # scale that fits the whole photo on screen
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0  # zoom factor and view origin (original px)
        self.sel = set()
        self.drag = None                             # (x0, y0, x1, y1) in original px while drawing
        self.mouse = (W / 2, H / 2)

    # coordinate transforms -------------------------------------------------
    def s(self):
        return self.base * self.zoom

    def to_orig(self, px, py):
        return self.ox + px / self.s(), self.oy + py / self.s()

    def to_disp(self, x, y):
        return int((x - self.ox) * self.s()), int((y - self.oy) * self.s())

    def clamp_view(self):
        H, W = self.orig.shape[:2]
        vw, vh = VIEW_W / self.s(), VIEW_H / self.s()
        self.ox = min(max(0, self.ox), max(0, W - vw))
        self.oy = min(max(0, self.oy), max(0, H - vh))

    def zoom_at(self, factor, px, py):
        x, y = self.to_orig(px, py)
        self.zoom = min(8.0, max(1.0, self.zoom * factor))
        self.ox, self.oy = x - px / self.s(), y - py / self.s()
        self.clamp_view()

    # rendering ------------------------------------------------------------
    def render(self):
        H, W = self.orig.shape[:2]
        x0, y0 = int(self.ox), int(self.oy)
        x1 = min(W, int(self.ox + VIEW_W / self.s()) + 1); y1 = min(H, int(self.oy + VIEW_H / self.s()) + 1)
        region = self.orig[y0:y1, x0:x1]
        interp = cv2.INTER_AREA if self.s() < 1 else cv2.INTER_LINEAR
        view = cv2.resize(region, None, fx=self.s(), fy=self.s(), interpolation=interp)
        canvas = np.zeros((VIEW_H + 28, VIEW_W, 3), np.uint8)
        dx, dy = self.to_disp(x0, y0)
        h, w = view.shape[:2]
        h, w = min(h, VIEW_H - dy), min(w, VIEW_W - dx)
        canvas[dy:dy + h, dx:dx + w] = view[:h, :w]
        for i, r in enumerate(self.boxes.rows):
            bx, by, bw, bh = (int(r[k]) for k in ("x", "y", "w", "h"))
            p0, p1 = self.to_disp(bx, by), self.to_disp(bx + bw, by + bh)
            c = SELECTED if i in self.sel else COLORS[r["label"]]
            cv2.rectangle(canvas, p0, p1, c, 2)
            cv2.putText(canvas, f"{r['idx']}{r['label'][:1]}", (p0[0], max(12, p0[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        if self.drag:
            p0, p1 = self.to_disp(*self.drag[:2]), self.to_disp(*self.drag[2:])
            cv2.rectangle(canvas, p0, p1, (255, 255, 0), 2)
        n = len(self.boxes.rows); lab = sum(1 for r in self.boxes.rows if r["label"])
        status = (f"{self.boxes.photo}  boxes={n} labeled={lab}  zoom={self.zoom:.1f}x  selected={len(self.sel)}"
                  f"  {'UNSAVED (S to save)' if self.boxes.dirty else 'saved'}   d=delete m=merge x/o/j=label u=undo q=quit")
        cv2.putText(canvas, status, (6, VIEW_H + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    # mouse ----------------------------------------------------------------
    def on_mouse(self, event, px, py, flags, _):
        self.mouse = (px, py)
        x, y = self.to_orig(px, py)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.press = (x, y, px, py, bool(flags & cv2.EVENT_FLAG_SHIFTKEY))
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_LBUTTONDOWN and getattr(self, "press", None):
            x0, y0, px0, py0, _ = self.press
            if abs(px - px0) > 5 or abs(py - py0) > 5:
                self.drag = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        elif event == cv2.EVENT_LBUTTONUP and getattr(self, "press", None):
            x0, y0, px0, py0, shift = self.press
            self.press = None
            if self.drag:                                            # finished drawing a new box
                bx0, by0, bx1, by1 = self.drag
                self.drag = None
                if bx1 - bx0 > 4 and by1 - by0 > 4:
                    self.sel = {self.boxes.add(bx0, by0, bx1 - bx0, by1 - by0)}
            else:                                                    # plain click: select
                hit = self.boxes.hit(x, y)
                if hit is None:
                    self.sel = set()
                elif shift:
                    self.sel ^= {hit}
                else:
                    self.sel = {hit}
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom_at(1.25 if flags > 0 else 0.8, px, py)

    # main loop ------------------------------------------------------------
    def run(self, on_save):
        win = f"edit {self.boxes.photo}"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.on_mouse)
        quit_armed = False
        while True:
            cv2.imshow(win, self.render())
            k = cv2.waitKey(30) & 0xFF
            if k == 255:
                continue
            quit_armed = quit_armed and k == ord("q")
            if k == ord("q"):
                if not self.boxes.dirty or quit_armed:
                    break
                print("unsaved edits: press S to save, or q again to discard"); quit_armed = True
            elif k == ord("S"):
                on_save(); print("saved")
            elif k == ord("d"):
                self.boxes.delete(self.sel); self.sel = set()
            elif k == ord("m"):
                new = self.boxes.merge(self.sel)
                self.sel = {new} if new is not None else set()
            elif k in (ord("x"), ord("o"), ord("j")):
                self.boxes.set_label(self.sel, {"x": "X", "o": "O", "j": "junk"}[chr(k)])
            elif k == ord("u"):
                self.boxes.undo(); self.sel = set()
            elif k in (ord("+"), ord("=")):
                self.zoom_at(1.25, *self.mouse)
            elif k == ord("-"):
                self.zoom_at(0.8, *self.mouse)
            elif k == ord("0"):
                self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0
            elif k in (ord("i"), ord("k"), ord("h"), ord("l")):
                step = 0.2 * VIEW_W / self.s()
                self.ox += {"h": -step, "l": step}.get(chr(k), 0)
                self.oy += {"i": -step, "k": step}.get(chr(k), 0)
                self.clamp_view()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", help="photo stem, e.g. photo1")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/crops"))
    args = ap.parse_args()

    manifest = args.out / "manifest.csv"
    rows = load_manifest(manifest)
    mine = [r for r in rows if r["photo"] == args.photo]
    others = [r for r in rows if r["photo"] != args.photo]
    if not mine:
        raise SystemExit(f"no rows for {args.photo} in {manifest}")
    orig = cv2.imread(str(args.raw / f"{args.photo}.jpg"))
    if orig is None:
        raise SystemExit(f"could not read {args.raw / (args.photo + '.jpg')}")

    boxes = Boxes(mine, args.photo)
    ed = Editor(orig, boxes)
    ed.run(on_save=lambda: save_all(manifest, others, boxes, orig, args.out))
    if boxes.dirty:
        print("quit without saving: edits discarded")


if __name__ == "__main__":
    main()
