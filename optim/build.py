from optim.optimizer.optim import get_optimizer
from optim.scheduler import get_scheduler
from optim.loss.contrastive_loss import LOSS_REGISTRY

def build_optim(cfg, params, total_steps):
    optimizer = get_optimizer(cfg, params)
    scheduler = get_scheduler(cfg, optimizer, total_steps)
     
    if 'retrieval' in cfg.model.loss.lower():
        loss = LOSS_REGISTRY.get(cfg.model.loss)()
    else:
        loss = LOSS_REGISTRY.get(cfg.model.loss)(cfg.model.base_modality)
    
    return loss, optimizer, scheduler