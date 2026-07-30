### Update paths & training parameters

train.py
```
### EDIT THESE TO FIT YOUR DATASET AND TRAINING PARAMETERS ###
NUM_CLASSES = 1
BATCH_SIZE = 8
EPOCHS = 600
EVAL_PERIOD = 100           # ~5 epochs * (number training images / BATCH_SIZE)
CHECKPOINT_PERIOD = 1000
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
### END OF EDITS ###
```

### Arguments

```
--version: version of the dataset
--dataset_name: name of the dataset
--config: path to the previously trained config file (optional)
--weights: path to the pre-trained weights file (optional)
```

`--dataset_name` should match the folder name in `_data/`.

`--version` should match the version in `_data/{dataset_name}/coco/train_{version}` and `_data/{dataset_name}/coco/val_{version}`.


### Start training
#### When pulling pre-trained config file from detectron2 model zoo:
```
python train.py \
  --version {version} \
  --dataset_name {dataset_name}
```

Example:
```
python train.py \
  --version v1 \
  --dataset_name screw
```

#### When continuing with your own pre-trained weights:
```
python train.py \
  --version {version} \
  --dataset_name {dataset_name} \
  --config {config_file} \
  --weights {weights}
```

Example:
```
python train.py \
  --version v1 \
  --dataset_name screw \
  --config _model/screw/v1/config.yaml \
  --weights _model/screw/v1/model_best.pth
```

## Inference

Run a trained model on an image/video file, a directory of images/videos, or a live
webcam feed. Annotated output is saved to `_output` (webcam feed is preview-only and
is not saved).

### Arguments

```
--config: path to a local config.yaml from a previously trained model, or a detectron2
          model zoo config name (e.g. COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml)
--weights: path to model weights. Required when --config is a local config.yaml.
           When --config is a model zoo name, defaults to that backbone's pretrained
           COCO weights, or can point to a locally trained .pth using that architecture.
--webcam: run live detection on a webcam (optionally pass a camera index, default: 0)
--path: path to an input image/video file, or a directory of images/videos
--output_dir: directory to save annotated output (default: _output)
--score_thresh: minimum score threshold for detections (default: 0.5)
--nms_thresh: non-maximum suppression IoU threshold; overlapping boxes above this are
              merged/suppressed. Lower = stricter (fewer overlapping boxes kept),
              higher = more permissive (default: 0.5)
--min_size_test: size of the shortest image side used at test time (INPUT.MIN_SIZE_TEST).
                 Defaults to the value from --config.
--max_size_test: maximum size of the longest image side used at test time
                 (INPUT.MAX_SIZE_TEST). Defaults to the value from --config.
--class_names: class names in label order, used to annotate detections and to infer
               NUM_CLASSES when using a model zoo config with custom-trained weights.
               Not needed when using default COCO-pretrained weights.
--color: draw all detections in a single fixed RGB color (0-255 each), e.g. --color 0 255 0,
         instead of random per-instance colors. Avoids the flickering/strobing effect random
         colors cause on video output.
--no_display: do not show a live preview window while processing
--cpu: force inference on CPU even if a GPU is available
```

Exactly one of `--webcam` or `--path` is required.

### Run on an image, video, or directory
```
python inference.py \
  --config _model/screw/v1/config.yaml \
  --weights _model/screw/v1/model_best.pth \
  --path _input \
  --class_names screw
```

### Run on a webcam
```
python inference.py \
  --config _model/screw/v1/config.yaml \
  --weights _model/screw/v1/model_best.pth \
  --webcam \
  --class_names screw
```
