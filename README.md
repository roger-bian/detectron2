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
python train.py --version {version} --dataset_name {dataset_name}
```

Example:
```
python train.py --version v1 --dataset_name screw
```

#### When continuing with your own pre-trained weights:
```
python train.py --version {version} --dataset_name {dataset_name} --config {config_file} --weights {weights}
```

Example:
```
python train.py --version v1 --dataset_name screw --config _model/screw/v1/config.yaml --weights _model/screw/v1/model_best.pth
```