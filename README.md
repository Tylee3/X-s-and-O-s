# CS221 mini-project: X vs O whiteboard classifier

Environment: `~/.venvs/rva312/bin/python` (has OpenCV 5, PyTorch 2.14, torchvision 0.29).
Run everything from this folder.

## Target: the class playground

The in-class test uses https://github.com/supertweety/model-playground. The website does its own
preprocessing (resize to 64x64 with no smoothing, luminosity grayscale, scale to [-1, 1]) and
feeds a float32 `[1,1,64,64]` tensor to an uploaded `.onnx` model that must return `[1,2]` raw
logits ordered (O, X). We vendor its Python preprocessing (`scripts/playground_lib/`) and
train on exactly that representation (`scripts/contract.py`). Colour never reaches the model,
so Step 1's min(R,G,B) trick cannot be used behind this interface; the local-background
subtraction can, and is built INTO every model (`models.BackgroundNormalize`) so it ships
inside the `.onnx`.

## Step 1 — segmentation

```bash
PY=~/.venvs/rva312/bin/python
$PY scripts/segment.py data/raw/photo1.jpg data/raw/photo3.jpg data/raw/photo4.jpg data/raw/photo5.jpg data/raw/photo7.jpg data/raw/photo8.jpg data/raw/photo9.jpg data/raw/photo10.jpg data/raw/photo11.jpg --out data/crops
$PY scripts/segment.py data/raw/photo2.jpg data/raw/photo6.jpg --out data/crops --thresh 25 --append
$PY scripts/edit_boxes.py photo1
```

Per photo in `data/crops/<photo>/`: `<photo>_NNN.png` crops, `_overlay.jpg` (numbered boxes),
`_sheet.jpg` (contact sheet), `_mask.png`. `manifest.csv` has one row per crop. `edit_boxes.py`
fixes boxes (click/shift-click select, `d` delete, `m` merge, drag to draw, `x`/`o`/`j` label,
`S` save). Ink detection: min(R,G,B), subtract a morphological-closing background estimate,
threshold, small dilation, connected components, filter by size and detected surface.

Photo pairs 2/6 and 4/5 are the same drawings re-shot with flash: they must share a split.

## Step 2 — dataset, preprocessing, augmentation

```bash
$PY scripts/dataset.py            # writes the split column: train 1,2,6,8,9,10 / val 3,4,5,7 / test 11
$PY scripts/preview_preproc.py    # data/preview_preproc.jpg: raw | browser input | model view | augmentations
```

Training-only augmentation on the colour crop: colour/brightness jitter; ZOOM-OUT to 35-100% of
the frame with random placement and 0.75-1.33 aspect squash (the website never crops and stretches
photos into a square: the first model called every small O an X); flip, rotate 30, perspective;
random source resolution so the browser's no-smoothing resize aliases every way; then the exact
contract conversion; then stroke thicken/thin, white erasing patches (glare), noise. Each step is
guarded: if the symbol vanished (local darkness peak < 25/255) the draw is retried.

## Step 3 — manual perceptron

```bash
$PY scripts/features.py
$PY scripts/perceptron_manual.py --show-mistakes     # edit WEIGHTS / BIAS at the top
```

Features = region ink density / average density on the model view: centre, ring, diagonal,
corners (useless: padding). centre +1, ring -1, bias -0.3. Fails on flat ovals; no single line fixes them.

## Steps 4 and 5 — MLP and CNN

```bash
$PY scripts/train.py --model mlp
$PY scripts/train.py --model cnn
$PY scripts/train.py --model mlp --no-augment    # overfitting ablation
```

Both: BackgroundNormalize front block, cross-entropy over 2 logits, Adam 1e-3, weight decay
1e-4, dropout 0.3, 60 epochs, keep best-val epoch. MLP 4096-128-64-2. CNN three conv+pool
blocks (16, 32, 64 channels) 64->8 px, fc 64, fc 2.

## Step 6 — comparison, export, live test

```bash
$PY scripts/compare.py --test              # results/: results.md, accuracy.png, confusion_*.png, curves, mistakes
$PY scripts/export_playground.py           # artifacts/{perceptron,mlp,cnn}.onnx + website-path accuracy
$PY scripts/live_demo.py --model cnn       # webcam rehearsal, m cycles CNN/MLP/perceptron
```

Upload `artifacts/cnn.onnx` (or the others) on the playground website. `results/professor_questions.md`
is the rehearsal list. Earlier results on the 32x32 ink pipeline (val 98.9% for both trained models)
are in git history before the playground rework.
