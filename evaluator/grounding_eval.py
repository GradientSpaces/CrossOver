import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
from omegaconf import DictConfig
from accelerate import Accelerator

from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator
from . import eval_utils

@EVALUATOR_REGISTRY.register()
class GroundingEval(BaseEvaluator):
    """Evaluator for grounding tasks that calculates top-k error metrics."""
    
    def __init__(self, cfg: DictConfig, accelerator: Accelerator, **kwargs: Any) -> None:
        """Initialize the grounding evaluator with configuration and accelerator."""
        self.task_name = cfg.task.name
        
        if 'scene' in self.task_name.lower():
            self.eval_func = eval_utils.calculate_topK_err_batch
        elif 'object' in self.task_name.lower():
            self.eval_func = eval_utils.calculate_topK_err
        else:
            raise NotImplementedError
        
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__ / self.task_name
        super().__init__(cfg, accelerator, **kwargs)
        
        self.eval_dict['err_top1'] = []
        self.eval_dict['err_top3'] = []
        
        for modality_combination in self.modality_combinations:
            src_modality, ref_modality = modality_combination
            
            self.eval_dict[src_modality + '_' + ref_modality + '_err_top1'] = []
            self.eval_dict[src_modality + '_' + ref_modality + '_err_top3'] = []
        
        self.eval_dict['target_metric'] = []
        
    def batch_metrics(self, data_dict: Dict[str, Any]) -> Dict[str, float]:
        """Calculate top-1 and top-3 error metrics for each modality combination in a batch."""
        metrics = {}
        output_embeddings = data_dict['embeddings']
        
        for modality_combination in self.modality_combinations:
            src_modality, ref_modality = modality_combination      
            
            mask = torch.logical_and(data_dict['masks'][src_modality], data_dict['masks'][ref_modality])
            
            if mask.sum() == 0:
                continue
            
            a_embed = output_embeddings[src_modality]
            b_embed = output_embeddings[ref_modality]
            
            if 'scene' in self.task_name.lower():
                assert len(a_embed.shape) == 3 and len(b_embed.shape) == 3
            
            err_top1, err_top3 = self.eval_func(a_embed, b_embed, mask, k=3)
                        
            metrics[src_modality + '_' + ref_modality + '_err_top1'] = err_top1.item()
            metrics[src_modality + '_' + ref_modality + '_err_top3'] = err_top3.item()
            
        all_top1_metric = [v for k, v in metrics.items() if '_err_top1' in k]
        all_top3_metric = [v for k, v in metrics.items() if '_err_top3' in k]

        metrics['target_metric'] = float(sum(all_top1_metric)) / len(all_top1_metric)
        metrics['err_top1'] = float(sum(all_top1_metric)) / len(all_top1_metric)
        metrics['err_top3'] = float(sum(all_top3_metric)) / len(all_top3_metric)
        
        return metrics