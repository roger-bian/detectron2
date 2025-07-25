### Update paths & training parameters

screw\detectron2_training_{dataset_name}.py
```
### EDIT THESE TO FIT YOUR DATASET AND TRAINING PARAMETERS ###
dataset_name = "screw" # name of dataset -- can be anything
dataset_train = "E:/paloma/task17/screw_head/coco/train_coco" # folder path to training dataset
dataset_valid = "E:/paloma/task17/screw_head/coco/valid_coco" # folder path to validation dataset

NUM_CLASSES = 1
BATCH_SIZE = 8
EPOCHS = 300
EVAL_PERIOD = 100

# model_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" ### Use this model for bounding box detection
model_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" ### Use this model for instance segmentation

output_dir = "E:/detectron2/model/screw_v1/" # output directory to save model weights
### END OF EDITS ###
```


### Start training
```
python screw\detectron2_training_{dataset_name}.py
```

#### Example
```
python screw\detectron2_training_screw_v1.py
```