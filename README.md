# detectron2 instance segmentation

Train and run [detectron2](https://github.com/facebookresearch/detectron2)
Mask R-CNN instance-segmentation models on custom datasets. The workflow is:
annotate in [labelme](https://github.com/wkentaro/labelme) → convert to COCO
format → train → run inference on images, video, or a webcam.

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
    ├── labelme/{train,val}/ # annotations + source images
    └── coco/{train,val}_{version}/
        ├── JPEGImages/     # or PNGImages/, per --image_format
        ├── Visualization/  # annotated previews, for eyeballing the conversion
        └── annotations.json
_model/{dataset_name}/{version}/   # training output: config.yaml, weights, metrics
_input/                     # images/videos to run inference on
_output/                    # annotated inference results
settings.yaml               # per-run settings for train.py and inference.py
src/trainer.py              # MyTrainer: validation-loss hook, best-checkpoint + early stop
src/settings.py             # settings.yaml loader + validation
labelme2coco/               # labelme → COCO converter, see labelme2coco/README.md
inference.ipynb             # notebook equivalent of inference.py
train.py, inference.py      # entry points
```

`_data/_{dataset_name}_example/` is an empty template — copy it to start a new
dataset. Everything under `_data/`, `_model/`, `_input/`, and `_output/` is
gitignored.

## Prepare a dataset

### 1. Annotate

Annotate images in labelme and save the JSON next to each image, split into
`_data/{dataset_name}/labelme/train/` and `.../val/`. The converter reads pixels
from `imageData` when present and otherwise falls back to the `imagePath` file
beside the JSON, so either labelme save style works.

### 2. Write a label list

Create `_data/{dataset_name}/labels.txt`, one label per line, with `__ignore__`
first:

```
__ignore__
ear
frame
```

Line order fixes the category IDs, so use the same file for both splits. Keeping
this per dataset (rather than editing the shared `labelme2coco/labels.txt`) is
the recommended convention — one shared file cannot serve two datasets with
different labels.

### 3. Convert to COCO

Run once per split:

```
python ./labelme2coco/labelme2coco.py \
  --input_dir ./_data/smm4-1/labelme/train \
  --output_dir ./_data/smm4-1/coco/train_v1 \
  --labels ./_data/smm4-1/labels.txt

python ./labelme2coco/labelme2coco.py \
  --input_dir ./_data/smm4-1/labelme/val \
  --output_dir ./_data/smm4-1/coco/val_v1 \
  --labels ./_data/smm4-1/labels.txt
```

Each run writes `JPEGImages/`, `Visualization/`, and `annotations.json`. Pass
`--noviz` to skip the visualization renders. Shapes whose label is absent from
`labels.txt` are silently dropped, so check the annotation count if it looks
low. See [labelme2coco/README.md](labelme2coco/README.md), and
[Known issues](#known-issues) for the pre-existing-output-directory error.

Add `--image_format png` to write `PNGImages/*.png` instead. PNG is lossless,
so it preserves PNG sources exactly where the default JPEG re-encode does not,
at roughly 10x the disk space. Either way the annotations file is
`annotations.json`, and `Visualization/` stays JPEG.

## Training

### Settings

All per-run settings live in the `train:` section of `settings.yaml`, each key
documented inline:

```
dataset_name:      dataset folder name under _data/
version:           dataset version, matching _data/{dataset_name}/coco/{train,val}_{version}
num_classes:       number of labels in labels.txt, excluding __ignore__
batch_size:        images per training batch
epochs:            passes over the training set
eval_period:       evaluate the validation set every N iterations
checkpoint_period: save a checkpoint every N iterations
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
`model_best.pth` whenever `segm/AP` improves, stopping early after 6 evaluations
with no gain of at least 0.002.

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

**labelme2coco refuses an existing `--output_dir`.** It exits with
`Output directory already exists` — but the dataset template ships
`coco/{train,val}_v1/.gitkeep`, so those directories exist from the start.
Convert to a scratch directory and move the contents in, or use a version that
does not exist yet.

**`labelme2coco/tools/resize_annotated_coco.py` needs `albumentations`,** which
is not in the dependency list. It is left out deliberately: it pulls
`opencv-python-headless`, which would install a second `cv2` alongside
`opencv-python`.

**`image_id` assignment is filesystem-order dependent.** labelme2coco globs
input files unsorted, so re-converting the same data can assign different image
IDs. Harmless for training; relevant if you diff two `annotations.json` files.
