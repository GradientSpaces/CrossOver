import os.path as osp
from safetensors.torch import load_file
import torch
from pathlib import Path
from datetime import timedelta
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from tqdm import tqdm
import numpy as np
import itertools 
from typing import Dict, Any, List
from omegaconf import DictConfig

from evaluator import eval_utils
from common import load_utils, misc
from data.build import build_dataloader
from model.build import build_model
from .build import EVALUATION_REGISTRY

@EVALUATION_REGISTRY.register()
class ObjectRetrieval():
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        set_seed(cfg.rng_seed)
        self.logger = get_logger(__name__)
        self.mode = cfg.mode
        
        key = "val"
        self.data_loader = build_dataloader(cfg, split=key, is_training=False)
        self.model = build_model(cfg)
        
        task_name = misc.rgetattr(cfg, "task.name")
        task_config = misc.rgetattr(cfg.task, task_name)
        self.dataset_name = misc.rgetattr(task_config, key)[0]
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]
        
        self.accelerator = Accelerator(kwargs_handlers=kwargs)
        
        # Accelerator preparation
        self.model, self.data_loader = self.accelerator.prepare(self.model, self.data_loader)
        
        # Load from ckpt
        self.ckpt_path = Path(cfg.task.get(cfg.task.name).ckpt_path)
        self.load_from_ckpt()
        
        if 'instance_baseline' in cfg.task.get(cfg.task.name).ckpt_path:
            assert cfg.dataloader.batch_size == 1, 'Cannot Run Object Level Grounding on Batched Scenes!'
            self.grounding_level = 'object'
        
        elif 'instance_crossover' in cfg.task.get(cfg.task.name).ckpt_path:
            self.grounding_level = 'scene'
        else:
            raise NotImplementedError
        
        self.modalities = cfg.task.get(cfg.task.name).modalities
        
        modality_combinations = list(itertools.combinations(self.modalities, 2))
        same_element_combinations = [(item, item) for item in self.modalities]
        modality_combinations.extend(same_element_combinations)
        self.modality_combinations = modality_combinations
        
        self.metrics = {}
        for modality_combination in self.modality_combinations:
            src_modality, ref_modality = modality_combination
            self.metrics[src_modality + '_' + ref_modality] = {}
            self.metrics[src_modality + '_' + ref_modality]['instance'] = {'recall_top1' : 0, 'recall_top3' : 0, 'count' : 0}
            self.metrics[src_modality + '_' + ref_modality]['scene'] = {'r@25' : 0, 'r@50' : 0, 'r@75' : 0, 'r@100' : 0, 'count' : 0}
             
    def forward(self, data_dict: Dict[str, Any]) ->  Dict[str, Any]:
        return self.model(data_dict)

    def prepareForObjectGrounding(self, data_dict: Dict[str, Any]) ->  Dict[str, Any]:
        object_data_dict = {}
        object_data_dict['inputs'] = data_dict['objects']['inputs']
        object_data_dict['masks']  = data_dict['masks']
        object_data_dict['label_ids'] = data_dict['label_ids']
        object_data_dict['scan_id'] = data_dict['scan_id']
        object_data_dict['scene_label'] = data_dict['scene_label']
        object_data_dict['scene_masks'] = data_dict['scene_masks']
        
        return object_data_dict

    @torch.no_grad()
    def inference_step(self) -> Dict[str, Any]:
        self.model.eval()
        loader = self.data_loader
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        self.logger.info('Running validation...')
        
        outputs = []
        for iter, data_dict in enumerate(loader):
            if self.grounding_level == 'object': # instance baseline
                data_dict = self.prepareForObjectGrounding(data_dict)
                data_dict = load_utils.get_first_index_batch(data_dict)
                data_dict = self.forward(data_dict)
                
                output = { 'scan_id' : data_dict['scan_id'], 'label': data_dict['scene_label'], 'scene_masks' : {}}
                output['label_ids'] = data_dict['label_ids']
                for modality in data_dict['embeddings']:
                    output[modality] = { }
                        
                    output[modality]['mask'] = data_dict['masks'][modality]
                    output[modality]['embedding'] = data_dict['embeddings'][modality]
                    output[modality]['scene_embedding'] = torch.mean(output[modality]['embedding'][output[modality]['mask'].to(torch.bool)], dim=0).reshape(-1,)
                    output['scene_masks'][modality] = (torch.sum(output[modality]['mask']) > 0).to(torch.bool)
                
                outputs.append(output)
            
            if self.grounding_level == 'scene': # instance crossover
                data_dict = self.forward(data_dict)
                num_scans = data_dict['embeddings']['rgb'].shape[0]
                for idx in range(num_scans):
                    output = { 'scan_id' : data_dict['scan_id'][idx], 'label': data_dict['scene_label'][idx], 'scene_masks' : {}}
                    output['label_ids'] = data_dict['label_ids'][idx]
                    output['object_ids'] = data_dict['objects']['object_ids'][idx]
                    
                    if self.dataset_name == 'Scan3R':
                        output['label_ids_rio_subset_mask'] = data_dict['label_ids_rio_subset_mask'][idx] 
                    
                    for modality in data_dict['embeddings']:
                        output[modality] = { }
                            
                        output[modality]['mask'] = data_dict['masks'][modality][idx]
                        output[modality]['embedding'] = data_dict['embeddings'][modality][idx]
                        output[modality]['scene_embedding'] = torch.mean(output[modality]['embedding'][output[modality]['mask'].to(torch.bool)], dim=0).reshape(-1,)
                        output['scene_masks'][modality] = (torch.sum(output[modality]['mask']) > 0).to(torch.bool)
                    
                    outputs.append(output)
            pbar.update(1)
        
        return outputs
    
    def eval(self, outputs: List[Dict[str, Any]]) -> None:
        for output_dict in outputs:
            for modality_combination in self.modality_combinations:
                src_modality, ref_modality = modality_combination
                mask = torch.logical_and(output_dict[src_modality]['mask'], output_dict[ref_modality]['mask'])
                
                a_embed = output_dict[src_modality]['embedding']
                b_embed = output_dict[ref_modality]['embedding']
                
                if mask.sum() == 0: 
                    continue
                
                a_embed = a_embed[mask]
                b_embed = b_embed[mask]
                
                sim = torch.softmax(a_embed @ b_embed.t(), dim=-1)
                rank_list = torch.argsort(1.0 - sim, dim = 1)
                top_k_indices = rank_list[:, :3]
                
                correct_index = torch.arange(a_embed.shape[0]).unsqueeze(1).to(top_k_indices.device)
                all_matches = top_k_indices == correct_index   
                 
                recall_top1 = all_matches[:, 0].float().mean()
                recall_top3 = all_matches.any(dim=1).float().mean()
                
                ratio = all_matches[:, 0].float().sum() / a_embed.shape[0]
                
                if ratio >= 0.75: 
                    self.metrics[src_modality + '_' + ref_modality]['scene']['r@75'] += 1
                if ratio >= 0.5: 
                    self.metrics[src_modality + '_' + ref_modality]['scene']['r@50'] += 1
                if ratio >= 0.25: 
                    self.metrics[src_modality + '_' + ref_modality]['scene']['r@25'] += 1
                
                self.metrics[src_modality + '_' + ref_modality]['instance']['recall_top1'] += recall_top1
                self.metrics[src_modality + '_' + ref_modality]['instance']['recall_top3'] += recall_top3
                
                self.metrics[src_modality + '_' + ref_modality]['instance']['count'] += 1
                self.metrics[src_modality + '_' + ref_modality]['scene']['count'] += 1        
        
        # Instance Matching Log
        self.logger.info('Instance Matching ---')
        for modality_combination in self.modality_combinations:
            src_modality, ref_modality = modality_combination
            count = self.metrics[src_modality + '_' + ref_modality]['instance']['count']
            if count == 0: 
                continue
            message = src_modality + '_' + ref_modality 
            
            inst_recall_top1 = self.metrics[src_modality + '_' + ref_modality]['instance']['recall_top1'] / count * 100.
            inst_recall_top3 = self.metrics[src_modality + '_' + ref_modality]['instance']['recall_top3'] / count * 100.
            
            message += f'-> top1_recall: {inst_recall_top1:.2f} | top3_recall: {inst_recall_top3:.2f}'
            self.logger.info(message)
        
        # Scene Level Matching
        self.logger.info('Scene Level Matching ---')
        for modality_combination in self.modality_combinations:
            src_modality, ref_modality = modality_combination
            count = self.metrics[src_modality + '_' + ref_modality]['scene']['count']
            
            if count == 0: 
                continue
            
            message = src_modality + '_' + ref_modality 
            
            scene_recall_r25 = self.metrics[src_modality + '_' + ref_modality]['scene']['r@25'] / count * 100
            scene_recall_r50 = self.metrics[src_modality + '_' + ref_modality]['scene']['r@50'] / count * 100
            scene_recall_r75 = self.metrics[src_modality + '_' + ref_modality]['scene']['r@75'] / count * 100
            
            message += f'-> r@25: {scene_recall_r25:.2f} | r@50: {scene_recall_r50:.2f} | r@75: {scene_recall_r75:.2f}'
            
            self.logger.info(message)
    
    def scene_eval(self, output_dict: Dict[str, Any]) -> None:
        scan_data = np.array([{ 'scan_id': output_data['scan_id'], 'label' : output_data['label']} for output_data in output_dict])
        unique_labels = {data['label'] for data in scan_data}

        for src_modality, ref_modality in self.modality_combinations:
            src_embed = torch.stack([output_data[src_modality]['scene_embedding'] for output_data in output_dict])
            ref_embed = torch.stack([output_data[ref_modality]['scene_embedding']for output_data in output_dict])
        
            src_mask  = torch.stack([output_data['scene_masks'][src_modality] for output_data in output_dict])
            ref_mask  = torch.stack([output_data['scene_masks'][ref_modality] for output_data in output_dict])
            mask = torch.logical_and(src_mask, ref_mask).reshape(-1,)
            
            if mask.sum() == 0:
                continue
            
            src_embed = src_embed[mask]
            ref_embed = ref_embed[mask]
            
            scan_data_masked = scan_data[mask.cpu().numpy()]
            
            sim = torch.softmax(src_embed @ ref_embed.t(), dim=-1)
            rank_list = torch.argsort(1.0 - sim, dim = 1)
            top_k_indices = rank_list[:, :20]
            
            correct_index = torch.arange(src_embed.shape[0]).unsqueeze(1).to(top_k_indices.device)
            matches = top_k_indices == correct_index   
            
            recall_top1 = matches[:, 0].float().mean() * 100.
            recall_top5 = matches[:, :5].any(dim=1).float().mean() * 100.
            recall_top10 = matches[:, :10].any(dim=1).float().mean() * 100.
            recall_top20 = matches.any(dim=1).float().mean() * 100.
            
            message = f"{src_modality} -> {ref_modality}:" 
            self.logger.info(message)
            
            message = f'Recall: top1 - {recall_top1:.2f}, top5 - {recall_top5:.2f}, top10 - {recall_top10:.2f}, top20 - {recall_top20:.2f}'
            self.logger.info(message)
            
            # Temporal eval
            recall_top1, recall_top5, recall_top10 = eval_utils.evaluate_temporal_scene_matching(rank_list.cpu().numpy().tolist(), scan_data_masked, self.data_loader.dataset.get_temporal_scan_pairs())
            message = f'Temporal: top1 - {recall_top1:.2f}, top5 - {recall_top5:.2f}, top10 - {recall_top10:.2f}'
            self.logger.info(message)  
            
            if self.dataset_name == 'Scannet':
                st_recall_top1, st_recall_top5, st_recall_top10 = eval_utils.calculate_scene_label_recall(rank_list.cpu().numpy().tolist(), scan_data_masked)
                st_recall_top1 *= 100.
                st_recall_top5 *= 100.
                st_recall_top10 *= 100.
                
                message  =  f"Category: top1 - {st_recall_top1:.2f}, top5 - {st_recall_top5:.2f}, top10 - {st_recall_top10:.2f}"
                self.logger.info(message)     
                
                ic_recall_top1, ic_recall_top3, ic_recall_top5 = eval_utils.evaluate_intra_category_scene_matching(scan_data_masked, src_embed, ref_embed, unique_labels)
                message  =  f"Intra-Category: top1 - {ic_recall_top1:.2f}, top5 - {ic_recall_top3:.2f}, top10 - {ic_recall_top5:.2f}"
                self.logger.info(message)  
    
    def load_from_ckpt(self) -> None:
        """Load model weights from checkpoint."""
        if self.ckpt_path.exists():
            self.logger.info(f"Loading from {self.ckpt_path}")
            # Load model weights from safetensors files
            ckpt = osp.join(self.ckpt_path, 'model.safetensors')
            ckpt = load_file(ckpt,  device = str(self.accelerator.device))
            self.model.load_state_dict(ckpt)
            self.logger.info(f"Successfully loaded from {self.ckpt_path}")
        
        else:
            raise FileNotFoundError
    
    def run(self) -> None:
        """Execute the complete inference and evaluation pipeline."""
        # Inference Step
        output_dict = self.inference_step()
        
        # Temporal Evaluation
        if self.dataset_name == 'Scan3R' and self.grounding_level == 'scene':
            self.logger.info('Temporal Instance Matching Evaluation...')
            scene_pairs = self.data_loader.dataset.get_temporal_scan_pairs()
            message = eval_utils.evaluate_temporal_instance_matching(scene_pairs, output_dict) 
            self.logger.info(message)
        
        self.logger.info('Object Retrieval Evaluation...')
        # Object Retrieval Evaluation
        self.eval(output_dict)
        
        self.logger.info('Scene Retrieval Evaluation (Instance CrossOver)...')
        # Scene Retrieval Evaluation
        self.scene_eval(output_dict)