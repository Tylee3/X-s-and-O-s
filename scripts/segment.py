"""
segment.py -- Step 1: cut a raw whiteboard/paper photo into one small image per X / O.

Usage:
    python scripts/segment.py data/raw/photo1.jpg [more photos...] --out data/crops

For every input photo this writes, under --out/<photo_stem>/:
    <stem>_NNN.png      one square crop per detected shape (NNN = 3-digit index)
    <stem>_overlay.jpg  the photo with every crop's box + index drawn on it
    <stem>_sheet.jpg    a contact sheet of all crops, for fast label spot-checking
    <stem>_mask.png     the binary "ink" mask (debug: see what the threshold caught)
and appends one row per crop to --out/manifest.csv (with an empty `label` column).

WHY THIS APPROACH (short version, the long one is in the chat):
  Marker strokes are thin, dark-ish lines on a light surface, but lighting varies
  wildly between photos.  So instead of asking "is this pixel dark?" we ask
  "is this pixel much darker than its *local* surroundings?"  That question has
  the same answer in a dim photo and a flash-lit one.
"""
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Tunable parameters.  Everything is expressed relative to image size so the same
# numbers work for a 5712-px-wide photo and a 2268-px-wide one.
# --------------------------------------------------------------------------- #
WORK_LONG_SIDE = 2400      # we do all detection on a copy resized to this long side (speed vs thin strokes)
CLOSE_FRAC     = 0.020     # background-estimate kernel = 2% of long side (must exceed stroke width)
FULL_FRAME     = False     # --full-frame: skip surface detection
INK_THRESH     = 35        # min "darker than background" (0..255) to count as ink; ghosts are ~10-20
JOIN_FRAC      = 0.005     # dilation radius to bridge small gaps in a stroke (0.5% of long side).
                           # Keep this SMALL: the two strokes of an X cross anyway; a big radius
                           # glued neighbouring shapes together in our first run.
MIN_INK_PX     = 40        # drop blobs with fewer ink pixels than this (dust, JPEG noise)
MIN_SIDE_FRAC  = 0.007     # drop blobs whose box is smaller than 0.7% of long side (~17 px: dots)
MAX_SIDE_FRAC  = 0.35      # drop blobs bigger than 35% of long side (frame, TV, shadows)
PAD_FRAC       = 0.18      # padding around each box before cropping (shape shouldn't touch the edge)
ASPECT_FLAG    = 1.8       # flag boxes taller/wider than this ratio: probably two shapes merged
FILL_FLAG      = 0.015     # flag boxes where ink covers < 1.5% of the box: frame edge / smear
SURFACE_MIN_FRAC = 0.10    # the detected board/paper must cover >= 10% of the photo, else use whole photo
SURFACE_INSET  = 0.012     # shrink the detected surface by 1.2% per side to exclude its frame


def load_and_resize(path: Path):
    """Load the photo and make a smaller working copy.  Returns (orig, work, scale)."""
    orig = cv2.imread(str(path))
    if orig is None:
        raise SystemExit(f"could not read {path}")
    long_side = max(orig.shape[:2])
    scale = WORK_LONG_SIDE / long_side if long_side > WORK_LONG_SIDE else 1.0
    work = cv2.resize(orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else orig.copy()
    return orig, work, scale


def ink_channel(bgr: np.ndarray) -> np.ndarray:
    """Darkest of the three colour channels at each pixel.

    A blue marker is dark in the RED channel, an orange marker is dark in the BLUE
    channel, black is dark everywhere.  Taking the per-pixel minimum makes every
    marker colour look roughly as dark as black ink, which plain grayscale does not
    (grayscale of light-blue is quite bright).
    """
    return bgr.min(axis=2)


def darkness_vs_background(ink: np.ndarray, long_side: int) -> np.ndarray:
    """Black top-hat: (local background) - (pixel).

    A morphological CLOSING with a kernel wider than a stroke fills the strokes in
    with the surrounding surface brightness, i.e. it estimates what the board would
    look like with nothing written on it.  Subtracting the real image from that
    estimate leaves only "things darker than their neighbourhood" -- the strokes --
    and cancels out gradients from lighting, glare fall-off, and camera vignetting.
    """
    k = int(CLOSE_FRAC * long_side) | 1                      # force odd kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    return cv2.subtract(background, ink)                      # saturates at 0, never negative


def ink_mask(darkness: np.ndarray) -> np.ndarray:
    """Threshold the darkness map into a binary mask and clean up speckle."""
    _, mask = cv2.threshold(darkness, INK_THRESH, 255, cv2.THRESH_BINARY)
    # NOTE: no morphological "opening" here.  At working resolution a marker stroke
    # is only 1-2 px wide, and an opening deletes anything thinner than its kernel --
    # it wiped out most strokes in our first run.  Speckle is removed later by the
    # minimum-size filter on connected components instead, which can't hurt strokes.
    return mask


def find_shapes(mask: np.ndarray, long_side: int):
    """Group nearby strokes into shapes and return one bounding box per shape.

    The two strokes of an X often don't touch, and an O may have a gap, so we
    DILATE the mask first (grow every stroke outward) so nearby strokes merge into
    one blob.  Connected-components labelling then gives us one label per blob.
    The bounding box is measured on the ORIGINAL (un-dilated) ink pixels of that
    blob so the padding isn't inflated by the dilation.
    """
    r = int(JOIN_FRAC * long_side) | 1
    joined = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    n, labels, _stats, _cent = cv2.connectedComponentsWithStats(joined, connectivity=8)

    shapes = []
    ink_ys, ink_xs = np.nonzero(mask)                         # coordinates of real ink pixels
    ink_labels = labels[ink_ys, ink_xs]                        # which blob each ink pixel belongs to
    for lab in range(1, n):                                    # label 0 is background
        sel = ink_labels == lab
        if not sel.any():
            continue
        xs, ys = ink_xs[sel], ink_ys[sel]
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        shapes.append(dict(x=int(x0), y=int(y0), w=int(x1 - x0 + 1), h=int(y1 - y0 + 1), ink=int(sel.sum())))
    return shapes


def find_surface(work: np.ndarray):
    """Locate the whiteboard / sheet of paper: the largest BRIGHT region in the photo.

    Blur heavily so strokes and glare specks disappear, then Otsu's threshold splits
    the image into "bright" and "not bright" automatically (it picks the cut that best
    separates the two peaks of the brightness histogram).  The biggest bright blob is
    the drawing surface.  We return its bounding box, shrunk a little so the metal
    frame around a whiteboard is excluded.  Falls back to the whole image if nothing
    convincing is found (e.g. the board fills the entire frame).
    """
    H, W = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    k = int(0.02 * max(H, W)) | 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=4)
    if n <= 1:
        return (0, 0, W, H), False
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[best]
    if area < SURFACE_MIN_FRAC * H * W:
        return (0, 0, W, H), False
    dx, dy = int(SURFACE_INSET * w), int(SURFACE_INSET * h)
    return (x + dx, y + dy, w - 2 * dx, h - 2 * dy), True

def filter_and_flag(shapes, work_shape, surface):
    """Reject obvious junk, attach human-readable warning flags to the survivors."""
    H, W = work_shape[:2]
    long_side = max(H, W)
    sx, sy, sw, sh = surface
    min_ink = MIN_INK_PX
    min_side, max_side = MIN_SIDE_FRAC * long_side, MAX_SIDE_FRAC * long_side
    kept = []
    for s in shapes:
        side = max(s["w"], s["h"])
        if s["ink"] < min_ink or side < min_side:
            continue                                            # dots, dust, JPEG noise
        if side > max_side:
            continue                                            # whiteboard frame, TV, big shadows
        cx, cy = s["x"] + s["w"] / 2, s["y"] + s["h"] / 2
        if not (sx <= cx <= sx + sw and sy <= cy <= sy + sh):
            continue                                            # centre is off the board/paper: TV, easel...
        flags = []
        aspect = max(s["w"], s["h"]) / max(1, min(s["w"], s["h"]))
        fill = s["ink"] / (s["w"] * s["h"])
        if aspect > ASPECT_FLAG:
            flags.append("elongated")                           # probably 2 shapes merged, or a smear
        if fill < FILL_FLAG:
            flags.append("sparse")                              # thin line across a big box: frame/smear
        if s["x"] <= sx + 2 or s["y"] <= sy + 2 or s["x"] + s["w"] >= sx + sw - 2 or s["y"] + s["h"] >= sy + sh - 2:
            flags.append("touches_edge")                        # runs into the frame / photo border
        s.update(aspect=round(aspect, 2), fill=round(fill, 3), flags="|".join(flags))
        kept.append(s)
    # Sort reading order (top-to-bottom in rows, then left-to-right) so indices are easy to find.
    row_h = 0.08 * long_side
    kept.sort(key=lambda s: (round((s["y"] + s["h"] / 2) / row_h), s["x"]))
    return kept


def square_crop(orig: np.ndarray, box, scale: float) -> np.ndarray:
    """Cut a padded SQUARE crop from the full-res photo around a work-image box.

    Square, because every downstream model wants a fixed aspect ratio and we do not
    want to squash tall O's into circles.  Padding so the stroke never touches the
    crop border (the CNN's edge filters behave badly there).  If the padded square
    runs off the photo we pad with the local median colour rather than black.
    """
    x, y, w, h = (box[k] / scale for k in ("x", "y", "w", "h"))
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * PAD_FRAC)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    H, W = orig.shape[:2]
    # amount we'd fall off each edge
    pl, pt, pr, pb = max(0, -x0), max(0, -y0), max(0, x1 - W), max(0, y1 - H)
    patch = orig[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if any((pl, pt, pr, pb)):
        fill = tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0))
        patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=fill)
    return patch


def photo_quality(work: np.ndarray, darkness: np.ndarray, mask: np.ndarray) -> dict:
    """Numbers that tell us whether this photo is trustworthy for segmentation."""
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    ink_vals = darkness[mask > 0]
    return dict(
        median_brightness=int(np.median(gray)),
        glare_pct=round(100 * float((gray >= 250).mean()), 2),      # blown-out pixels
        dark_pct=round(100 * float((gray <= 60).mean()), 2),        # near-black pixels (TV, shadows)
        median_ink_contrast=int(np.median(ink_vals)) if ink_vals.size else 0,
    )


def contact_sheet(crops, labels_txt, tile=96, cols=12) -> np.ndarray:
    """Tile all crops with their index into one image for quick review."""
    if not crops:
        return np.full((tile, tile, 3), 255, np.uint8)
    rows = math.ceil(len(crops) / cols)
    sheet = np.full((rows * tile, cols * tile, 3), 255, np.uint8)
    for i, (c, t) in enumerate(zip(crops, labels_txt)):
        r, col = divmod(i, cols)
        thumb = cv2.resize(c, (tile - 4, tile - 4), interpolation=cv2.INTER_AREA)
        y, x = r * tile + 2, col * tile + 2
        sheet[y:y + tile - 4, x:x + tile - 4] = thumb
        cv2.putText(sheet, t, (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return sheet


def process(path: Path, out_root: Path, writer):
    stem = path.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    orig, work, scale = load_and_resize(path)
    long_side = max(work.shape[:2])
    ink = ink_channel(work)
    dark = darkness_vs_background(ink, long_side)
    mask = ink_mask(dark)
    surface, found = ((0, 0, work.shape[1], work.shape[0]), False) if FULL_FRAME else find_surface(work)
    shapes = filter_and_flag(find_shapes(mask, long_side), work.shape, surface)

    overlay = work.copy()
    sx, sy, sw, sh = surface
    cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 3)   # blue = detected surface
    crops, tags = [], []
    for i, s in enumerate(shapes):
        name = f"{stem}_{i:03d}.png"
        crop = square_crop(orig, s, scale)
        cv2.imwrite(str(out_dir / name), crop)
        crops.append(crop)
        tags.append(f"{i}{'!' if s['flags'] else ''}")
        color = (0, 140, 255) if s["flags"] else (0, 200, 0)          # orange = flagged, green = clean
        cv2.rectangle(overlay, (s["x"], s["y"]), (s["x"] + s["w"], s["y"] + s["h"]), color, 2)
        cv2.putText(overlay, str(i), (s["x"], max(12, s["y"] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        writer.writerow(dict(
            file=str((out_dir / name).relative_to(out_root)), photo=stem, idx=i,
            x=int(s["x"] / scale), y=int(s["y"] / scale), w=int(s["w"] / scale), h=int(s["h"] / scale),
            ink_px=s["ink"], aspect=s["aspect"], fill=s["fill"], flags=s["flags"], label="",
        ))

    cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
    cv2.imwrite(str(out_dir / f"{stem}_sheet.jpg"), contact_sheet(crops, tags), [cv2.IMWRITE_JPEG_QUALITY, 90])

    q = photo_quality(work, dark, mask)
    n_flag = sum(1 for s in shapes if s["flags"])
    print(f"{stem:8s} surface={'auto' if found else 'FULL'} crops={len(shapes):3d} flagged={n_flag:2d}  brightness={q['median_brightness']:3d} "
          f"glare={q['glare_pct']:5.2f}%  dark={q['dark_pct']:5.2f}%  ink_contrast={q['median_ink_contrast']:3d}")
    return q


def main():
    global INK_THRESH, FULL_FRAME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/crops"))
    ap.add_argument("--thresh", type=int, default=INK_THRESH,
                    help="min darkness-vs-background to count as ink (default %(default)s). "
                         "Lower (~25) for faded/dim photos, higher (~45) for glossy wrinkled paper.")
    ap.add_argument("--append", action="store_true", help="append to manifest.csv instead of overwriting")
    ap.add_argument("--full-frame", action="store_true",
                    help="skip board/paper detection and search the whole photo (dark boards fool the brightness-based detector)")
    args = ap.parse_args()
    INK_THRESH = args.thresh                                   # module-level knob, overridden per run
    FULL_FRAME = args.full_frame
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = args.out / "manifest.csv"
    fields = ["file", "photo", "idx", "x", "y", "w", "h", "ink_px", "aspect", "fill", "flags", "label"]
    mode = "a" if args.append and manifest.exists() else "w"
    with open(manifest, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for p in args.photos:
            process(p, args.out, writer)
    print(f"\nmanifest -> {manifest}   (fill in the `label` column with X / O / junk)")


if __name__ == "__main__":
    main()
