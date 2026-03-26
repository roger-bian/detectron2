import os
import shutil
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

from src.trainer import MyTrainer



### EDIT THESE TO FIT YOUR DATASET AND TRAINING PARAMETERS ###
version = "v2"
dataset_name = "side_panel" # name of dataset -- can be anything
dataset_train = f"/home/user/code/roger-bian/detectron2/_data/side_panel/coco/train_{version}/_resized" # folder path to training dataset
dataset_valid = f"/home/user/code/roger-bian/detectron2/_data/side_panel/coco/val_{version}/_resized" # folder path to validation dataset

NUM_CLASSES = 2
BATCH_SIZE = 8
EPOCHS = 600
EVAL_PERIOD = 100           # ~5 epochs * (number training images / BATCH_SIZE)
CHECKPOINT_PERIOD = 1000

# model_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" ### Use this model for bounding box detection
model_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" ### Use this model for instance segmentation

output_dir = f"/home/user/code/roger-bian/detectron2/_model/side_panel/{version}" # output directory to save model weights
### END OF EDITS ###



def register_dataset():
    register_coco_instances(f"{dataset_name}_train", {}, f"{dataset_train}/annotations.json", dataset_train)
    register_coco_instances(f"{dataset_name}_val", {}, f"{dataset_valid}/annotations.json", dataset_valid)


if __name__ == '__main__':
    register_dataset()

    dataset = DatasetCatalog.get(f"{dataset_name}_train")

    num_images = len(dataset)
    ITERATIONS_PER_EPOCH = num_images/BATCH_SIZE
    NUM_ITERATIONS = int(EPOCHS * ITERATIONS_PER_EPOCH)

    # 2. Set Up Configurations for Training
    cfg = get_cfg()
    # cfg.set_new_allowed(True)

    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{dataset_name}_val",)
    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD # evaluate validation set after this number of iterations
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    # cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)  # Pre-trained weights
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
    cfg.SOLVER.MAX_ITER = NUM_ITERATIONS  # Adjust as needed
    cfg.SOLVER.STEPS = []  # Learning rate decay
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD # save checkpoint after this number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES  # Replace with your number of classes
    

    # Output directory to save model weights
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)

    # setup logger
    setup_logger(output=cfg.OUTPUT_DIR)

    # save config.yaml
    config_save_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(cfg.dump())
    print(f"Configuration saved to: {config_save_path}")


    # 3. Training the Model
    print(f"Epochs{EPOCHS} has {NUM_ITERATIONS} iterations.")
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


    # # Save the model weights after training
    # torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
    # print(f"Model saved to {cfg.OUTPUT_DIR}/model_final.pth")