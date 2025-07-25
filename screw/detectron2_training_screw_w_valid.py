import os
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import time
import datetime
import logging
import numpy as np

if __name__ == '__main__':
    class LossEvalHook(HookBase):
        def __init__(self, eval_period, model, data_loader):
            self._model = model
            self._period = eval_period
            self._data_loader = data_loader
        
        def _do_loss_eval(self):
            # Copying inference_on_dataset from evaluator.py
            total = len(self._data_loader)
            num_warmup = min(5, total - 1)
                
            start_time = time.perf_counter()
            total_compute_time = 0
            losses = []
            for idx, inputs in enumerate(self._data_loader):            
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                start_compute_time = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=5,
                    )
                loss_batch = self._get_loss(inputs)
                losses.append(loss_batch)
            mean_loss = np.mean(losses)
            self.trainer.storage.put_scalar('validation_loss', mean_loss)
            comm.synchronize()

            return losses
                
        def _get_loss(self, data):
            # How loss is calculated on train_loop 
            metrics_dict = self._model(data)
            metrics_dict = {
                k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                for k, v in metrics_dict.items()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())
            return total_losses_reduced
            
            
        def after_step(self):
            next_iter = self.trainer.iter + 1
            is_final = next_iter == self.trainer.max_iter
            if is_final or (self._period > 0 and next_iter % self._period == 0):
                self._do_loss_eval()
            self.trainer.storage.put_scalars(timetest=12)
    
    
    class MyTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
                        
        def build_hooks(self):
            hooks = super().build_hooks()
            hooks.insert(-1,LossEvalHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg,True)
                )
            ))
            return hooks



    
    # 1. Register the Dataset (replace paths with your dataset paths)
    dataset_name = "screw"
    def register_dataset():
        register_coco_instances(f"{dataset_name}_train", {}, "E:/paloma/task17/screw_head/coco/train_coco/annotations.json", "E:/paloma/task17/screw_head/coco/train_coco")
        register_coco_instances(f"{dataset_name}_val", {}, "E:/paloma/task17/screw_head/coco/valid_coco/annotations.json", "E:/paloma/task17/screw_head/coco/valid_coco")

    register_dataset()

    dataset = DatasetCatalog.get(f"{dataset_name}_train")

    num_images = len(dataset)
    BATCH_SIZE = 8
    ITERATIONS_PER_EPOCH = num_images/BATCH_SIZE
    EPOCHS = 300
    NUM_ITERATIONS = int(EPOCHS * ITERATIONS_PER_EPOCH)
    NUM_CLASSES = 1
    EVAL_PERIOD = 100

    # 2. Set Up Configurations for Mask R-CNN
    cfg = get_cfg()
    # cfg.set_new_allowed(True)
    model_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

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
    cfg.SOLVER.CHECKPOINT_PERIOD = EVAL_PERIOD # save checkpoint after this number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES  # Replace with your number of classes
    

    # Output directory to save model weights
    cfg.OUTPUT_DIR = "E:/detectron2/model/screw_v1/"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Epochs{EPOCHS} has {NUM_ITERATIONS} iterationssteps")
    # 3. Training the Model
    trainer = MyTrainer(cfg)
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
    # evaluator = COCOEvaluator(f"{dataset_name}_val", cfg, False, output_dir="./model/glove_v3/evaluation")
    # test_result = trainer.test(cfg, model=trainer.model, evaluators=[evaluator])
    # print(test_result)


