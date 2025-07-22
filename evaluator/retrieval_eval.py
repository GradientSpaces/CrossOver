import torch
from pathlib import Path
from itertools import combinations
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
import yaml
from accelerate import Accelerator

from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator
from . import eval_utils

@EVALUATOR_REGISTRY.register()
class RetrievalEval(BaseEvaluator):
    def __init__(self, cfg: DictConfig, accelerator: Accelerator, **kwargs: Any) -> None:
        """Initialize the retrieval evaluator."""
        self.task_name = cfg.task.name
        
        self.eval_func = eval_utils.calculate_topK_err
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__ / self.task_name
        super().__init__(cfg, accelerator, **kwargs)
        
        self.modalities = self.cfg.task.get(self.cfg.task.name).scene_modalities
        self.modalities.append('object')
        self.modality_combinations = list(combinations(list(self.modalities), 2))
        
        self.eval_dict['err_top1'] = []
        self.eval_dict['err_top5'] = []
        
        for modality_combination in self.modality_combinations:
            src_modality, ref_modality = modality_combination
            self.eval_dict[src_modality + '_' + ref_modality + '_err_top1'] = []
            self.eval_dict[src_modality + '_' + ref_modality + '_err_top5'] = []
        
        self.eval_dict['target_metric'] = []
        self.freeze_object_enc = self.cfg.task.get(self.cfg.task.name).freeze_object_enc
        
        if not self.freeze_object_enc: 
            # self.grounding_config_path = self.cfg.task.get(self.cfg.task.name).scene_level_grounding_eval.config_path
            # with open(self.grounding_config_path, 'r') as f:
            #     grounding_config = yaml.safe_load(f)
            
            # grounding_eval_cfg = OmegaConf.create(grounding_config)
            self.grounding_eval = EVALUATOR_REGISTRY.get('GroundingEval')(cfg, accelerator, **kwargs)


    def batch_metrics(self, data_dict: Dict[str, Any]) -> Dict[str, float]:
        """Calculate retrieval metrics for a batch of embeddings."""
        metrics = {}
        output_embeddings = data_dict['embeddings']
        
        for modality_combination in self.modality_combinations:
            src_modality, ref_modality = modality_combination      
            
            a_embed = output_embeddings[src_modality]
            b_embed = output_embeddings[ref_modality]      
            
            mask = torch.logical_and(data_dict['scene_masks'][src_modality], data_dict['scene_masks'][ref_modality]).reshape(-1, )
            if mask.sum() == 0:
                continue
                  
            assert a_embed.shape == b_embed.shape
            
            err_top1, err_top5 = self.eval_func(a_embed, b_embed, mask=None, label_ids=None, k = 5) 
            metrics[src_modality + '_' + ref_modality + '_err_top1'] = err_top1.item()
            metrics[src_modality + '_' + ref_modality + '_err_top5'] = err_top5.item()
        
        all_top1_metric = [v for k, v in metrics.items() if '_err_top1' in k]
        all_top5_metric = [v for k, v in metrics.items() if '_err_top5' in k]

        metrics['target_metric'] = float(sum(all_top1_metric)) / len(all_top1_metric)
        metrics['err_top1'] = float(sum(all_top1_metric)) / len(all_top1_metric)
        metrics['err_top5'] = float(sum(all_top5_metric)) / len(all_top5_metric)
        
        if not self.freeze_object_enc:
            instance_data_dict = {}
            instance_data_dict['embeddings'] = data_dict['object_modality_embeddings']
            instance_data_dict['masks'] = data_dict['masks']
            instance_metrics = self.grounding_eval.batch_metrics(instance_data_dict)
            metrics['instance_target_metric'] = instance_metrics['target_metric']
            metrics['instance_err_top1'] = instance_metrics['err_top1']
            metrics['instance_err_top3'] = instance_metrics['err_top3']
        
        return metrics