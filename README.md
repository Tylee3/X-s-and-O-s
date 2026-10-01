# CS221 mini-project: X vs O whiteboard classifier

Environment: `~/.venvs/rva312/bin/python` (has OpenCV 5, PyTorch 2.14, torchvision 0.29).
Run everything from this folder.

## Step 1 — segmentation (done, awaiting your review)

```bash
PY=~/.venvs/rva312/bin/python
# good-contrast photos with the default threshold, dim ones with a lower threshold
$PY scripts/segment.py data/raw/photo1.jpg data/raw/photo3.jpg data/raw/photo4.jpg data/raw/photo5.jpg --out data/crops
$PY scripts/segment.py data/raw/photo2.jpg data/raw/photo6.jpg --out data/crops --thresh 25 --append
```

Outputs, per photo, in `data/crops/<photo>/`:

| file | what it is |
|---|---|
| `<photo>_NNN.png` | one square crop per detected shape |
| `<photo>_overlay.jpg` | the photo with numbered boxes. Green = clean, orange = flagged, blue = detected board/paper |
| `<photo>_sheet.jpg` | contact sheet of all crops; `!` after a number = flagged |
| `<photo>_mask.png` | the binary ink mask (debug) |

`data/crops/manifest.csv` has one row per crop with an empty `label` column.

### Reviewing and labelling

1. Skim each `_sheet.jpg` first. Anything that is not a single clean X or O is `junk`.
2. Label with keypresses (writes into `manifest.csv` after every key, safe to quit any time):

```bash
$PY scripts/label.py                 # x / o / j / b=back / s=skip / q=quit
$PY scripts/label.py --only-flagged  # revisit only the suspicious ones
```

3. Duplicate photos: photo5 is the same sheet as photo4 (flash), photo6 is the same board as
   photo2 (flash). Same drawings, different lighting. They must go in the SAME train/val split
   later or validation accuracy will be inflated (handled in Step 2).

### Fixing wrong boxes (one shape split in two, two shapes in one box, junk)

```bash
$PY scripts/edit_boxes.py photo1
```

Click a box to select it, shift-click to add more. `d` deletes, `m` merges the selection into one
box, click-drag on empty board draws a new box, `x` / `o` / `j` labels the selection, `u` undoes,
scroll wheel or `+`/`-` zooms, `i k h l` pans, `0` resets the view. `S` (capital) saves: it rewrites
the manifest rows for that photo, deletes crops of removed boxes, cuts crops for new ones and redraws
the overlay and contact sheet. `q` quits (press twice to discard unsaved edits).
Box colours: green X, blue O, grey junk, orange unlabeled, red selected.

## Step 2 — dataset, preprocessing, augmentation

```bash
$PY scripts/dataset.py            # writes the `split` column into the manifest, prints the tiers
$PY scripts/preview_preproc.py    # data/preview_preproc.jpg: raw | ink | binary | 6 augmentations
```

Split is by PHOTO (train: 1,2,6,8,9,10; val: 3,4,5,7; test: 11) so a re-shot drawing can never
leak across tiers. Rows without an X/O label get an empty split and are excluded.

Preprocessing (`to_ink`): min(R,G,B) -> subtract morphological-closing background -> divide by
the crop's own peak -> 32x32. Background 0, stroke ~1, independent of marker colour and lighting.
`binarize=True` thresholds at 0.4 for the ablation.

Training-only augmentation: colour/brightness jitter on the raw crop (proves to_ink cancels it),
then flip, rotate +-30, translate 10%, scale 0.8-1.2, mild perspective, stroke thicken/thin,
small random erasing (glare gaps), Gaussian noise.

## Step 3 — manual perceptron

```bash
$PY scripts/features.py                              # per-class feature distributions (train)
$PY scripts/perceptron_manual.py --show-mistakes     # edit WEIGHTS / BIAS at the top of the file
```

Features (features.py): ink density in centre square / ring / diagonal band / corners, each divided
by the crop's average density. Corners turned out useless (padding). Weights centre +1, ring -1,
bias -0.3 give train 96.3% / val 91.4%. Failures: flat ovals (stroke crosses the centre) - not fixable
by any single line on these features.

## Step 4 — MLP

```bash
$PY scripts/train.py --model mlp                 # runs/mlp/: curves.png, confusion.png, mistakes_val.jpg
$PY scripts/train.py --model mlp --no-augment    # ablation: overfits (train 100%, val drops, val loss rises)
$PY scripts/train.py --model mlp --binarize      # ablation: 0/1 pixels
```

1024 -> 128 -> 64 -> 1, ReLU, dropout 0.3, BCE-with-logits, Adam lr 1e-3, weight decay 1e-4,
60 epochs, keep the best-val-accuracy epoch. Val 98.9% with augmentation, 95.7% without.

## Step 5 — CNN

```bash
$PY scripts/train.py --model cnn                 # runs/cnn/
$PY scripts/train.py --model cnn --no-augment    # ablation
```

models_cnn.py: conv(1->16,3x3) ReLU pool, conv(16->32,3x3) ReLU pool, flatten 2048, dropout 0.3,
fc 64, fc 1. 136k params, only 4.8k of them in the conv layers. Same data/loss/optimizer/script as the MLP.

## Step 6 — comparison

```bash
$PY scripts/compare.py --test      # results/: results.md, accuracy.png, confusion_*.png, curves_mlp_vs_cnn.png, mistakes_*.jpg
```

| classifier | train (510) | val (93) | test (49, photo11) |
|---|---|---|---|
| manual perceptron | 96.3% | 91.4% | 100% |
| MLP | 98.4% | 98.9% | 100% |
| CNN | 98.6% | 98.9% | 100% |

`results/professor_questions.md` is the list of questions to rehearse (questions only).
