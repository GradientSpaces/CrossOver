from tqdm import tqdm
from omegaconf import DictConfig

import torch
from trainer.build import TRAINER_REGISTRY
from trainer.build import BaseTrainer

@TRAINER_REGISTRY.register()
class GroundingTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        
    def train_step(self, epoch: int) -> None:
        self.model.train()
        loader = self.data_loaders["train"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process), desc=f"[Epoch {epoch + 1}/{self.epochs}]")
        
        for i, data_dict in enumerate(loader):
            with self.accelerator.accumulate(self.model):
                # forward
                data_dict = self.forward(data_dict)
                # calculate loss
                loss, loss_dict = self.loss(data_dict)
                # calculate evaluator
                metrics = self.evaluator['train'].batch_metrics(data_dict)
                self.backward(loss)
                self.global_step += 1
                
                log_dict = {'step': self.global_step}
                log_dict.update(loss_dict)
                log_dict.update(metrics)
                
                # optimize
                if self.global_step > 0 and self.global_step % 50 == 0:
                    self.log(log_dict, mode="train")
                    self.evaluator['train'].reset()
                
                pbar.update(1)
        
        self.scheduler.step()
        
    @torch.no_grad()
    def eval_step(self, epoch: int) -> bool:
        self.model.eval()
        loader = self.data_loaders["val"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            loss, losses = self.loss(data_dict)
            self.evaluator['val'].update(data_dict)
            pbar.update(1)
        is_best, results = self.evaluator['val'].record()
        if is_best:
            self.best_metric = results["target_metric"]
        self.log(results, mode="val")
        self.evaluator['val'].reset()
        return is_best

    def run(self) -> None:
        if self.mode == "train":
            start_epoch = self.exp_tracker.epoch
            self.global_step = start_epoch * len(self.data_loaders["train"])
            for epoch in range(start_epoch, self.epochs):
                self.exp_tracker.step()
                self.train_step(epoch)

                if self.epochs_per_eval and (epoch + 1) % self.epochs_per_eval == 0:
                    is_best = self.eval_step(epoch)
                    self.accelerator.print(f"[Epoch {epoch + 1}/{self.epochs}] finished eval, is_best: {is_best}")
                else:
                    is_best = False

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    if is_best:
                        self.save("best.pth")
                    if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                        self.save(f"ckpt_{epoch+1}.pth")
        self.accelerator.end_training()
