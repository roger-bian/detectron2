# detectron2 instance segmentation

Train and run [detectron2](https://github.com/facebookresearch/detectron2)
Mask R-CNN instance-segmentation models on custom datasets. The workflow is:
annotate in [labelme](https://github.com/wkentaro/labelme) → build a versioned
COCO dataset with one command → train → run inference on images, video, or a
webcam.

## Setup

Requires Python 3.10 and, for GPU training, a CUDA 12.8 capable GPU. The
detectron2 [prebuilt wheel](https://github.com/facebookresearch/detectron2/discussions/5200)
is built for `pt2.8.0cu128` and cp310 only, so neither the Python version nor
the torch/torchvision pins are free to move.

### uv (recommended)

```
uv sync
```

This creates `.venv` from `uv.lock` with the exact pinned versions. Then either
activate it (`source .venv/bin/activate`) or prefix the commands below with
`uv run` — e.g. `uv run python train.py`.

### pip

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### conda

```
conda env create -f environment.yml
conda activate detectron2
```

`pyproject.toml` is the canonical dependency list; `requirements.txt` and
`environment.yml` mirror it for non-uv setups.

## Repository layout

```
_data/                      # datasets — see _data/README.md
└── {dataset_name}/
    ├── labels.txt          # label list, "__ignore__" first
    ├── labelme/            # what you edit: images + labelme JSON
    │   ├── train/
    │   ├── val/
    │   └── test/           # optional
    └── coco/               # built by prepare_dataset.py, never edited by hand
        └── {split}_v{N}/
            ├── JPEGImages/     # or PNGImages/, per --image_format
            ├── Visualization/  # annotated previews, for eyeballing the conversion
            ├── annotations.json
            ├── manifest.json   # exactly which files (and hashes) went in
            └── labels.txt      # copy of the label list used
_model/{dataset_name}/{version}/   # training output: config.yaml, weights, metrics
_input/                     # images/videos to run inference on
_output/                    # annotated inference results
settings.yaml               # per-run settings for train.py and inference.py
src/trainer.py              # MyTrainer: validation-loss hook, best-checkpoint + early stop
src/settings.py             # settings.yaml loader + validation
labelme2coco/               # labelme → COCO converter used by prepare_dataset.py
inference.ipynb             # notebook equivalent of inference.py
prepare_dataset.py          # labelme splits → next COCO dataset version
train.py, inference.py      # entry points
```

`_data/_{dataset_name}_example/` is an empty template — copy it to start a new
dataset. Everything under `_data/`, `_model/`, `_input/`, and `_output/` is
gitignored.

## Prepare a dataset

### 1. Create the dataset folder

Copy the template and rename it:

```
cp -r "_data/_{dataset_name}_example" _data/my_dataset
```

It contains `labels.txt` and empty `labelme/train/`, `val/`, and `test/`
folders. A dataset can also be nested, e.g. `_data/smm4-1/water_drops`; the
dataset name is then `smm4-1/water_drops` everywhere.

### 2. Write the label list

Edit `_data/my_dataset/labels.txt`: one label per line, `__ignore__` first:

```
__ignore__
ear
frame
```

Line order fixes the category IDs (`ear` = 0, `frame` = 1), so only ever append
to it. `train.num_classes` is the number of lines after `__ignore__`.

### 3. Split the images and annotate

Put each image in the split it belongs to — `labelme/train/`, `labelme/val/`,
and optionally `labelme/test/` — then annotate in labelme, saving each JSON next
to its image. The folder an image is in is its split, so to move an image to
another split, move its JSON with it. You can also annotate first and split
afterwards; only where the files are when you build counts.

- Use polygon, rectangle, or circle shapes. Shapes with the same label and
  `group_id` become one object (e.g. an object split in two by something in
  front of it).
- A **background image** (one with nothing to find, used to suppress false
  positives) still needs a JSON: open it in labelme and save with no shapes.
  An image without a JSON counts as not annotated yet and is left out.
- Label a shape `__ignore__` to keep it in the JSON but out of the dataset.
- Folders starting with `_` (e.g. `labelme/_unused/`) are ignored.

### 4. Build a dataset version

```
python prepare_dataset.py my_dataset
```

This checks every split first and writes nothing if any check fails:

| check | result |
|---|---|
| `train/` or `val/` missing or with no annotated images | error |
| the same image (by content) in two splits | error — leakage |
| JSON whose image is missing or unreadable | error |
| shape the converter cannot handle (`mask`, `line`, `point`, a polygon under 3 points, ...) | error |
| label not in `labels.txt` (would be dropped) | warning |
| image with no JSON (not annotated yet, left out) | warning |
| the same image twice in one split | warning |

It then prints a per-split summary (images, background images, objects per
label) and converts every split into `coco/{split}_v{N}/`, where N is one more
than the highest version already there, so `train_vN`, `val_vN`, and `test_vN`
always belong together. Existing versions are never touched. If a conversion
fails partway, nothing is left behind.

Each version folder gets a `manifest.json` recording the git commit, the label
list, and every source file with its SHA-256, so you can always tell exactly
what a model was trained on, and a copy of `labels.txt`.

| option | effect |
|---|---|
| `--dry_run` | run the checks and print the summary, write nothing |
| `--strict` | treat warnings as errors, e.g. for a final dataset that must be fully annotated |
| `--image_format png` | write lossless `PNGImages/*.png` instead of `JPEGImages/*.jpg`; keeps PNG sources pixel-for-pixel at ~10x the disk space |
| `--noviz` | skip the `Visualization/` previews |

### 5. Point training at it

The script ends by printing the values to put in the `train:` section of
`settings.yaml`:

```
dataset_name: my_dataset
version: v1
num_classes: 2
```

It does not edit `settings.yaml` itself, so building a version never changes
what the next training run uses. `train.py` reads `train_vN` and `val_vN`;
`test_vN` is for your own evaluation.

## Training

### Settings

All per-run settings live in the `train:` section of `settings.yaml`, each key
documented inline:

```
dataset_name:      dataset folder name under _data/
version:           dataset version built by prepare_dataset.py, e.g. v3 (reads coco/{train,val}_v3)
num_classes:       number of labels in labels.txt, excluding __ignore__
batch_size:        images per training batch
epochs:            passes over the training set
eval_period:       evaluate the validation set every N iterations
checkpoint_period: save a checkpoint every N iterations
early_stop_patience: evaluations without segm/AP gain before stopping (null = run the full schedule)
config_file:       model zoo backbone to train from
config:            resume from a previous run's config.yaml (null = use config_file)
weights:           pre-trained weights (null = model zoo COCO weights)
```

`config_file` chooses the architecture; `config`/`weights` resume from a model
you already trained.

The learning rate is **not** a setting — `train.py` derives it from
`batch_size` by the linear scaling rule (`0.02` at batch 16, detectron2's recipe
for this config), so changing the batch size does not silently change how hard
the model trains. LR decay milestones are likewise fractions of the total run
(0.67 and 0.89) rather than fixed iteration counts. Both are constants at the
top of `train.py`, and the derived values are printed at startup and saved into
the run's `config.yaml`.

Batch size is limited by GPU memory: roughly **1026 MiB per image plus 428 MiB**
fixed, measured on this dataset. Batch 4 needs ~4.5 GB; batch 8 needs ~8.6 GB
and will not fit on an 8 GB card. Lower `batch_size` if training stops with
`torch.OutOfMemoryError`.

### Run

Edit the `train:` section of `settings.yaml`, then:

```
python train.py
```

To run a different configuration without editing the tracked file, copy it and
pass the copy as the first argument:

```
python train.py experiments/lr_test.yaml
```

### Output

Everything lands in `_model/{dataset_name}/{version}/`: `config.yaml`, periodic
`model_{iteration}.pth` checkpoints, `metrics.json`, and TensorBoard event
files. `MyTrainer` also evaluates every `eval_period` iterations and writes
`model_best.pth` whenever `segm/AP` improves, stopping early after
`early_stop_patience` evaluations with no gain of at least 0.002. With
`early_stop_patience: null` the full schedule runs (so both LR drops happen) and
detectron2 also writes `model_final.pth` at the last iteration.

This directory is created with `exist_ok=False`, so **re-running with the same
`version` fails** rather than overwriting. Bump the version or delete the
directory first.

## Inference

Run a trained model on an image/video file, a directory of images/videos, or a
live webcam feed. Annotated output is saved to `_output` (webcam feed is
preview-only and is not saved).

### Settings

All per-run settings live in the `inference:` section of `settings.yaml`, each
key documented inline:

```
config:         local config.yaml from a trained model, or a model zoo config name
weights:        model weights; required when config is a local config.yaml
path:           input image/video file, a directory of them, or `webcam`
output_dir:     where annotated output is saved
score_thresh:   minimum detection score
nms_thresh:     non-maximum suppression IoU threshold; lower = stricter
min_size_test:  shortest image side at test time (null = take from config)
max_size_test:  longest image side at test time (null = take from config)
class_names:    class names in label order, e.g. [mask, screw, hook]
color:          fixed [R, G, B] for all detections (null = random per instance)
mask_alpha:     mask fill opacity, 0-1; only applies when color is set
box_color:      fixed [R, G, B] for bounding boxes only
no_display:     do not show a live preview window
cpu:            force CPU even if a GPU is available
```

Set `path` to `webcam` to run live detection on a camera; you are prompted
for the device id, and nothing is written to `output_dir`.
Setting `color` avoids the flickering that random per-instance colors cause on
video output.

### Run

Point `config`, `weights`, and `class_names` at your model, set `path`, then:

```
python inference.py
```

For a file or directory:

```
config: _model/side_panel/v3/config.yaml
weights: _model/side_panel/v3/model_best.pth
class_names: [mask, screw, hook]
path: _input
```

For a camera, set `path: webcam`. You are then asked:

```
Enter webcam device id (0 is the default camera):
```

Press Enter for camera 0, or type an index. Webcam runs are preview-only —
nothing is written to `output_dir`. To skip the prompt in an unattended run,
pipe the index in: `echo 1 | python inference.py`.

As with training, a copy of the file can be passed as the first argument:

```
python inference.py experiments/webcam.yaml
```

## Known issues

**`labelme2coco/tools/resize_annotated_coco.py` needs `albumentations`,** which
is not in the dependency list. It is left out deliberately: it pulls
`opencv-python-headless`, which would install a second `cv2` alongside
`opencv-python`.
