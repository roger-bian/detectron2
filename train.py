import os
import argparse

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

from src.trainer import MyTrainer


### EDIT THESE TO FIT YOUR DATASET AND TRAINING PARAMETERS ###
NUM_CLASSES = 1
BATCH_SIZE = 8
EPOCHS = 600
EVAL_PERIOD = 100           # ~5 epochs * (number training images / BATCH_SIZE)
CHECKPOINT_PERIOD = 1000
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
### END OF EDITS ###


def main(args):
    dataset_train = f"_data/{args.dataset_name}/coco/train_{args.version}" # folder path to training dataset
    dataset_valid = f"_data/{args.dataset_name}/coco/val_{args.version}" # folder path to validation dataset
    register_coco_instances(
        f"{args.dataset_name}_train",
        {},
        f"{dataset_train}/annotations.json",
        dataset_train
    )
    register_coco_instances(
        f"{args.dataset_name}_val",
        {},
        f"{dataset_valid}/annotations.json",
        dataset_valid
    )
    dataset = DatasetCatalog.get(f"{args.dataset_name}_train")

    num_images = len(dataset)
    ITERATIONS_PER_EPOCH = num_images/BATCH_SIZE
    NUM_ITERATIONS = int(EPOCHS * ITERATIONS_PER_EPOCH)


    if args.config is not None:
        # use local pretrained config yaml
        config_file = args.config
    else:
        # get pre-trained from detectron2 model zoo
        config_file = model_zoo.get_config_file(CONFIG_FILE)

    if args.weights is not None:
        # use local pretrained weights
        weights = args.weights
    else:
        # get pre-trained from detectron2 model zoo
        weights = model_zoo.get_checkpoint_url(CONFIG_FILE)


    cfg = get_cfg()
    # cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.DATASETS.TRAIN = (f"{args.dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{args.dataset_name}_val",)
    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD # evaluate validation set after this number of iterations
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    # cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.MODEL.WEIGHTS = weights
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
    cfg.SOLVER.MAX_ITER = NUM_ITERATIONS  # Adjust as needed
    cfg.SOLVER.STEPS = []  # Learning rate decay
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD # save checkpoint after this number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES  # Replace with your number of classes
    

    # Output directory to save model weights
    cfg.OUTPUT_DIR = f"_model/{args.dataset_name}/{args.version}"
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        help="Version to train"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the previously trained config file."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to the pre-trained weights file."
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)