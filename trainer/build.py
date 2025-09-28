from datetime import timedelta
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from fvcore.common.registry import Registry
import torch
import numpy as np

from data.build import build_dataloader
from model.build import build_model
from optim.build import build_optim
from evaluator.build import build_eval
from common.load_utils import make_dir

TRAINER_REGISTRY = Registry("Trainer")

class Tracker():
    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize tracking state for training."""
        self.reset(cfg)

    def step(self) -> None:
        """Increment epoch counter by one."""
        self.epoch += 1

    def reset(self, cfg: OmegaConf) -> None:
        """Reset tracker state with configuration."""
        self.exp_name = f"{cfg.exp_dir.parent.name.replace(f'{cfg.name}', '').lstrip('_')}/{cfg.exp_dir.name}"
        self.epoch = 0
        self.best_result = -np.inf

    def state_dict(self) -> Dict[str, Any]:
        """Return tracker state as dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load tracker state from dictionary."""
        self.__dict__.update(state_dict)

@TRAINER_REGISTRY.register()
class BaseTrainer():
    def __init__(self, cfg: DictConfig) -> None:
        """Initialize trainer with configuration."""
        set_seed(cfg.rng_seed)
        
        self.global_step = 0
        self.epochs_per_eval = cfg.solver.get("epochs_per_eval", None)
        self.epochs_per_save = cfg.solver.get("epochs_per_save", None)
        
        self.logger = get_logger(__name__)
        self.mode = cfg.mode
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]
        
         # Initialize accelerator
        self.exp_tracker = Tracker(cfg)
        wandb_args = {"id": cfg.logger.run_id, "resume": cfg.resume}
        if not cfg.logger.get('autoname'):
            wandb_args["name"] = self.exp_tracker.exp_name
        
        gradient_accumulation_steps = cfg.solver.get("gradient_accumulation_steps", 1)
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs
        )
        
        self.accelerator.init_trackers(
                    project_name=cfg.name,
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) if not cfg.resume else None,
                    init_kwargs={"wandb": wandb_args}
                )
        
        keys = ["train", "val"]
        self.data_loaders = {key : build_dataloader(cfg, split=key, is_training=True) for key in keys}
        self.model = build_model(cfg)
        
        total_steps = (len(self.data_loaders["train"]) * cfg.solver.epochs) // gradient_accumulation_steps
        
        params = getattr(self.model, 'get_opt_params', list(self.model.parameters()))
        
        if not isinstance(params, list):
            params = params(cfg.solver.lr)
        
        self.loss, self.optimizer, self.scheduler = build_optim(cfg, [{'params' : params}], total_steps= total_steps)
        self.evaluator = build_eval(cfg, self.accelerator)
        
        self.epochs = cfg.solver.epochs
        self.total_steps = len(self.data_loaders["train"]) * cfg.solver.epochs
        self.grad_norm = cfg.solver.get("grad_norm")
        
        # Accelerator preparation
        self.model, self.loss, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.loss, self.optimizer, self.scheduler)
        for name, loader in self.data_loaders.items():
            if isinstance(loader, list):
                loader = self.accelerator.prepare(*loader)
            else:
                loader = self.accelerator.prepare(loader)
            self.data_loaders[name] = loader
        self.accelerator.register_for_checkpointing(self.exp_tracker)
        
        self.ckpt_path = Path(cfg.ckpt_path) if cfg.get("ckpt_path") else Path(cfg.exp_dir) / "ckpt" / "best.pth"
        if cfg.resume:
            self.resume()
    
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of data through the model."""
        return self.model(data_dict)

    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass and optimization step."""
        self.optimizer.zero_grad()
        self.accelerator.backward(loss, retain_graph=True)
        
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
    
    def log(self, results: Dict[str, Any], mode: str = "train") -> None:
        """Log training metrics and learning rates."""
        log_dict = {}
        for key, val in results.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            log_dict[f"{mode}/{key}"] = val
        if mode == "train":
            lrs = self.scheduler.get_lr()
            for i, lr in enumerate(lrs):
                log_dict[f"{mode}/lr/group_{i}"] = lr
        
        self.accelerator.log(log_dict, step=self.global_step)
    
    def save(self, name: str) -> None:
        """Save model checkpoint with given name."""
        make_dir(self.ckpt_path.parent)
        self.save_func(str(self.ckpt_path.parent / name))
    
    def resume(self) -> None:
        """Resume training from checkpoint if available."""
        if self.ckpt_path.exists():
            self.logger.info(f"Resuming from {str(self.ckpt_path)}")
            self.accelerator.load_state(str(self.ckpt_path))
            self.logger.info(f"Successfully resumed from {self.ckpt_path}")
        else:
            self.logger.info("training from scratch")
    
    def save_func(self, path: str) -> None:
        """Save model state to specified path."""
        self.accelerator.save_state(path)

def build_trainer(cfg: DictConfig) -> BaseTrainer:
    """Create trainer instance based on configuration."""
    return TRAINER_REGISTRY.get(cfg.trainer)(cfg)