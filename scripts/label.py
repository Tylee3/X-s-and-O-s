"""
label.py -- label the crops from segment.py with single keypresses.

Usage:
    python scripts/label.py                      # label everything still unlabeled
    python scripts/label.py --relabel            # revisit everything, including labeled crops
    python scripts/label.py --only-flagged       # only crops segment.py marked as suspicious

Keys:  x = X      o = O      j = junk (smudge, merged shapes, cut off, not a shape)
       b = back one      s = skip for now      q = quit
Progress is written to the manifest after every keypress, so quitting is always safe.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

KEYS = {ord("x"): "X", ord("o"): "O", ord("j"): "junk"}


def save(manifest: Path, rows, fields):
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/crops/manifest.csv"))
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--only-flagged", action="store_true")
    ap.add_argument("--size", type=int, default=420, help="display size in px")
    args = ap.parse_args()

    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    root = args.manifest.parent

    todo = [i for i, r in enumerate(rows)
            if (args.relabel or not r["label"]) and (not args.only_flagged or r["flags"])]
    if not todo:
        print("nothing to label"); return

    win = "label  (x / o / j=junk / b=back / s=skip / q=quit)"
    cv2.namedWindow(win)
    pos = 0
    while 0 <= pos < len(todo):
        r = rows[todo[pos]]
        img = cv2.imread(str(root / r["file"]))
        img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
        canvas = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        head = f"{pos + 1}/{len(todo)}  {r['file']}"
        sub = f"flags={r['flags'] or '-'}   current={r['label'] or '-'}"
        cv2.putText(canvas, head, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, sub, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, canvas)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("b"):
            pos = max(0, pos - 1)
        elif k == ord("s"):
            pos += 1
        elif k in KEYS:
            r["label"] = KEYS[k]
            save(args.manifest, rows, fields)
            pos += 1
    cv2.destroyAllWindows()

    counts = Counter(r["label"] or "unlabeled" for r in rows)
    print("labels so far:", dict(counts))


if __name__ == "__main__":
    main()
