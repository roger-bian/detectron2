import os

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

from src.trainer import MyTrainer
from src.settings import load_settings

### Training parameters live in the `train:` section of settings.yaml ###

# base_lr is derived from batch_size by the linear scaling rule: the learning
# rate is scaled in proportion to the batch, so changing batch_size does not
# silently change how hard the model trains. Expressed as a reference pair --
# LR_REFERENCE_LR at LR_REFERENCE_BATCH -- which is detectron2's own recipe for
# this config. (The previous hardcoded 0.00025 was ~40x below this and did not
# scale, so batch 8 trained at half the effective rate of batch 4.)
LR_REFERENCE_BATCH = 16
LR_REFERENCE_LR = 0.02

# Learning-rate decay milestones, as fractions of the total run rather than
# fixed iteration counts -- iterations depend on batch_size, so the old
# STEPS = [4800] landed at a different point in training for every batch size.
LR_DECAY_FRACTIONS = (0.67, 0.89)


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
    ITERATIONS_PER_EPOCH = num_images/args.batch_size
    NUM_ITERATIONS = int(args.epochs * ITERATIONS_PER_EPOCH)


    if args.config is not None:
        # use local pretrained config yaml
        config_file = args.config
    else:
        # get pre-trained from detectron2 model zoo
        config_file = model_zoo.get_config_file(args.config_file)

    if args.weights is not None:
        # use local pretrained weights
        weights = args.weights
    else:
        # get pre-trained from detectron2 model zoo
        weights = model_zoo.get_checkpoint_url(args.config_file)


    cfg = get_cfg()
    # cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.DATASETS.TRAIN = (f"{args.dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{args.dataset_name}_val",)
    cfg.TEST.EVAL_PERIOD = args.eval_period # evaluate validation set after this number of iterations
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    # cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.MODEL.WEIGHTS = weights
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = LR_REFERENCE_LR * args.batch_size / LR_REFERENCE_BATCH
    cfg.SOLVER.MAX_ITER = NUM_ITERATIONS  # Adjust as needed
    cfg.SOLVER.STEPS = [int(f * NUM_ITERATIONS) for f in LR_DECAY_FRACTIONS]
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period # save checkpoint after this number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes  # Replace with your number of classes
    
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative"
    cfg.INPUT.CROP.SIZE = [0.7, 0.7]

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
    print(f"Epochs{args.epochs} has {NUM_ITERATIONS} iterations.")
    print(
        f"BASE_LR {cfg.SOLVER.BASE_LR:g} derived from batch_size {args.batch_size} "
        f"({LR_REFERENCE_LR:g} at batch {LR_REFERENCE_BATCH}); "
        f"LR decay at iterations {list(cfg.SOLVER.STEPS)}"
    )
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    args = load_settings("train")
    main(args)
