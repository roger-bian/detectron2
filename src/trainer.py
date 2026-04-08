import os
import time
import datetime
import logging
import math
import numpy as np
import torch
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.events import EventStorage
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm

logger = logging.getLogger("detectron2")

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


class BestCheckpointAndEarlyStopHook(HookBase):
    def __init__(
        self,
        metric_name,
        eval_period=0,
        patience=5,
        min_delta=0.0,
        file_prefix="model_best",
    ):
        self.metric_name = metric_name
        self.eval_period = eval_period
        self.patience = patience
        self.min_delta = min_delta
        self.file_prefix = file_prefix

        self.best_value = -math.inf
        self.bad_epochs = 0

    def after_step(self):
        trainer = self.trainer
        next_iter = trainer.iter + 1

        if self.eval_period <= 0:
            return

        is_eval_step = (
            next_iter % self.eval_period == 0
            or next_iter == trainer.max_iter
        )
        if not is_eval_step:
            return

        storage = trainer.storage
        if self.metric_name not in storage.histories():
            return

        history = storage.history(self.metric_name)
        if not history.values():
            return

        current = history.latest()

        if current > self.best_value + self.min_delta:
            self.best_value = current
            self.bad_epochs = 0

            logger.info(
                f"[Best+EarlyStop] New best {self.metric_name}: {current:.4f}"
            )
            
            trainer.checkpointer.save(self.file_prefix)
            logger.info(
                f"[Best+EarlyStop] New best checkpoint saved: {self.file_prefix}.pth"
            )

        else:
            self.bad_epochs += 1
            logger.info(
                f"[Best+EarlyStop] No improvement "
                f"({self.bad_epochs}/{self.patience})"
            )

        if self.bad_epochs >= self.patience:
            logger.info(
                f"[Best+EarlyStop] EARLY STOPPING triggered. "
                f"Best {self.metric_name}: {self.best_value:.4f}"
            )
            trainer.should_stop = True


class MyTrainer(DefaultTrainer):
    def train(self):
        self.should_stop = False  # <-- EARLY STOP FLAG

        logger.info(
            f"Starting training from iteration {self.start_iter}"
        )

        self.iter = self.start_iter
        self.max_iter = self.cfg.SOLVER.MAX_ITER

        with EventStorage(self.start_iter) as self.storage:
            self.before_train()

            for self.iter in range(self.start_iter, self.max_iter):
                if self.should_stop:
                    logger.info(
                        f"Early stopping at iter {self.iter}"
                    )
                    break

                self.before_step()
                self.run_step()
                self.after_step()

            # important: match TrainerBase semantics
            self.iter += 1
            self.after_train()

        if (
            len(self.cfg.TEST.EXPECTED_RESULTS)
            and comm.is_main_process()
        ):
            self._last_eval_results = getattr(self, "_last_eval_results", None)
            return self._last_eval_results
    

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    
    def build_hooks(self):
        hooks = super().build_hooks()
        
        # validation loss hook
        hooks.insert(
            -1,
            LossEvalHook(
                eval_period=self.cfg.TEST.EVAL_PERIOD,
                model=self.model,
                data_loader=build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg, True)
                )
            )
        )

        # best model based on segmentation AP
        hooks.insert(
            -1,
            BestCheckpointAndEarlyStopHook(
                metric_name="segm/AP",      # bbox/AP, segm/AP
                eval_period=self.cfg.TEST.EVAL_PERIOD,
                patience=6,
                min_delta=0.002,
                file_prefix="model_best"
            )
        )

        return hooks