import os
import cv2
import torch
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from datetime import datetime


if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 1. Register the Dataset (replace paths with your dataset paths)

    dataset_name = "glove"
    def register_dataset():
        register_coco_instances(f"{dataset_name}_train", {}, "E:/toda/training/glove_v2/annotations/train.json", "E:/toda/training/glove_v2/train")
        register_coco_instances(f"{dataset_name}_val", {}, "E:/toda/training/glove_v2/annotations/valid.json", "E:/toda/training/glove_v2/valid")

    register_dataset()

    dataset = DatasetCatalog.get(f"{dataset_name}_train")

    num_images = len(dataset)
    BATCH_SIZE = 8
    ITERATIONS_PER_EPOCH = num_images/BATCH_SIZE
    EPOCHS = 300
    NUM_ITERATIONS = int(EPOCHS * ITERATIONS_PER_EPOCH)
    NUM_CLASSES = 1


    # 2. Set Up Configurations for Mask R-CNN
    cfg = get_cfg()
    model_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{dataset_name}_val",)
    # cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)  # Pre-trained weights
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
    cfg.SOLVER.MAX_ITER = NUM_ITERATIONS  # Adjust as needed
    cfg.SOLVER.STEPS = []  # Learning rate decay
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES  # Replace with your number of classes

    # Output directory to save model weights
    cfg.OUTPUT_DIR = "E:/detectron2/model/glove/" + current_time

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Epochs{EPOCHS} has {NUM_ITERATIONS} iterationssteps")
    # 3. Training the Model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()



    # Save the model weights after training
    torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
    print(f"Model saved to {cfg.OUTPUT_DIR}/model_final.pth")
    config_save_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(cfg.dump())
    print(f"Configuration saved to: {config_save_path}")

    # After training, evaluate the model
    evaluator = COCOEvaluator(f"{dataset_name}_val", cfg, False, output_dir="./output4/")
    test_result = trainer.test(cfg, model=trainer.model, evaluators=[evaluator])
    print(test_result)


