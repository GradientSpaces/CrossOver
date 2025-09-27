import os.path as osp
import os
import argparse
from accelerate import Accelerator
from datetime import timedelta
import torch
from accelerate.utils import InitProcessGroupKwargs
import random
import numpy as np
import open3d as o3d
import logging as log
from PIL import Image
from torchvision import transforms as tvf
from typing import Dict, List, Tuple

import sys
sys.path.append(osp.abspath('.'))
from model.instance_crossover import InstanceCrossOverModel
from util import torch_util
from omegaconf import DictConfig

from single_inference.datasets.scannet_instance import ScannetInstanceInference
from single_inference.datasets.scan3r_instance import Scan3RInstanceInference
from single_inference.datasets.arkit_instance import ARKitScenesInstanceInference
from single_inference.datasets.multiscan_instance import MultiScanInstanceInference

DEFAULT_CONFIG = {
    'dataset': 'scannet',  # scannet, scan3r, arkitscenes, multiscan
    'data_dir': '/drive/datasets/Scannet',  # Update this with your data path
    'process_dir': '/drive/dumps/multimodal-spaces/preprocess_feats/Scannet',  # Update this with your processed data path
    'ckpt': '/drive/dumps/multimodal-spaces/runs/new_runs/instance_crossover_scannet+scan3r+multiscan+arkitscenes.pth',  # Update this with your model checkpoint
    'scan_id': 'scene0568_00',  # Default scan to search in
    'query_modality': 'point',  # point, rgb, referral
    'target_modality': 'referral',  # point, rgb, referral, cad
    'query_path': './demo_data/kitchen/scene.ply',  # Path to your query file
    'top_k': 5  
}
# =============================================================================

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

class InstanceRetrieval:
    """Simple instance retrieval within a single scene"""
    
    def __init__(self, args):
        self.args = args
        self.setup_model()
        
        # Setup image transforms
        self.image_transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
    
    def setup_model(self):
        """Initialize and load the model"""
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        kwargs = [init_kwargs]
        self.accelerator = Accelerator(kwargs_handlers=kwargs)
        
        # Convert args to DictConfig format expected by model
        model_args = DictConfig({
            'out_dim': self.args.out_dim,
            'input_dim_3d': self.args.input_dim_3d,
            'input_dim_2d': self.args.input_dim_2d,
            'input_dim_1d': self.args.input_dim_1d
        })
        
        self.model = InstanceCrossOverModel(model_args, self.accelerator.device)
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        self.model.to(self.model.device)
        torch_util.load_weights(self.model, self.args.ckpt, self.accelerator.device)
        
        log.info(f"Model loaded from {self.args.ckpt}")
    
    def setup_dataset(self, scan_id):
        """Setup single-scan dataset based on the dataset type"""
        if self.args.dataset == 'scannet':
            self.dataset = ScannetInstanceInference(self.args.data_dir, scan_id, shape_dir='/drive/datasets/Shapenet/ShapeNetCore.v2/')
        elif self.args.dataset == 'scan3r':
            self.dataset = Scan3RInstanceInference(self.args.data_dir, self.args.process_dir, scan_id)
        elif self.args.dataset == 'arkitscenes':
            self.dataset = ARKitScenesInstanceInference(self.args.data_dir, self.args.process_dir, scan_id)
        elif self.args.dataset == 'multiscan':
            self.dataset = MultiScanInstanceInference(self.args.data_dir, self.args.process_dir, scan_id)
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
    
    def encode_query(self, query_path: str, query_modality: str) -> torch.Tensor:
        """Encode query object and return embedding"""
        
        if query_modality == 'point':
            return self._encode_point_query(query_path)
        elif query_modality == 'rgb':
            return self._encode_rgb_query(query_path)
        elif query_modality == 'referral':
            return self._encode_referral_query(query_path)
        else:
            raise NotImplementedError(f'Query modality {query_modality} not implemented')
    
    def _encode_point_query(self, path: str) -> torch.Tensor:
        """Encode point cloud query - matches dataset approach with raw point cloud"""
        assert path.endswith('.ply'), 'Point Cloud Path should be a .ply file!'
        
        pcd = o3d.io.read_point_cloud(path) 
        points = np.asarray(pcd.points)
        
        # Send raw point cloud as list (like datasets) - model will handle sampling
        point_clouds = [points]  # List of raw point clouds
        point_masks = torch.ones(1, 1).bool()  # (1, 1)
        
        data_dict = {
            'objects': {
                'inputs': {'point': point_clouds}  # List format, not tensor
            },
            'masks': {'point': point_masks}
        }
        
        with torch.no_grad():
            embed = self.model(data_dict)['embeddings']['point'].cpu()
        
        return embed.squeeze()
    
    def _encode_rgb_query(self, path: str) -> torch.Tensor:
        """Encode RGB image query"""
        assert os.isfile(path), 'RGB Path should be an image file'
        
        image = Image.open(path)
        image = image.resize((224, 224), Image.BICUBIC)
        image_pt = self.image_transform(image).unsqueeze(0)  # (1, C, H, W)
        
        # Convert to model expected format: (batch_size, num_objects, num_views, C, H, W)
        rgb_data = image_pt.unsqueeze(0).unsqueeze(0)  # (1, 1, 1, C, H, W)
        rgb_masks = torch.ones(1, 1).bool()  # (1, 1)
        
        data_dict = {
            'objects': {
                'inputs': {'rgb': rgb_data}
            },
            'masks': {'rgb': rgb_masks}
        }
        
        with torch.no_grad():
            embed = self.model(data_dict)['embeddings']['rgb'].cpu()
        
        return embed.squeeze()
    
    def _encode_referral_query(self, path: str) -> torch.Tensor:
        """Encode text referral query"""
        if os.isfile(path):
            with open(path, 'r') as f:
                text = f.read().strip()
        else:
            text = path  # Assume path is the text itself
        
        data_dict = {'referral_texts': [[[text]]]}
        
        with torch.no_grad():
            embed = self.model(data_dict)['embeddings']['referral'].cpu()
        
        return embed.squeeze()
    
    def encode_scene(self, scan_id: str) -> Dict[str, torch.Tensor]:
        """Encode all objects in the scene and return embeddings by modality"""
        
        # Setup dataset for this specific scan
        self.setup_dataset(scan_id)
        
        # Get the data for this scan
        data_dict = self.dataset.get_data()
        
        
        with torch.no_grad():
            output = self.model(data_dict)
        
        # Extract embeddings and masks for each modality
        scene_embeddings = {}
        for modality in output['embeddings']:
            embeddings = output['embeddings'][modality].cpu()
            masks = data_dict['masks'][modality].cpu()
            
            # Remove batch dimension
            if len(embeddings.shape) == 3:
                embeddings = embeddings.squeeze(0)
            if len(masks.shape) == 2:
                masks = masks.squeeze(0)
            
            # Store embeddings and masks
            scene_embeddings[modality] = {
                'embeddings': embeddings,
                'masks': masks,
                'object_ids': data_dict.get('object_ids', list(range(embeddings.shape[0])))
            }
        
        return scene_embeddings
    
    def retrieve(
        self, 
        query_path: str, 
        query_modality: str, 
        scan_id: str, 
        target_modality: str, 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Main retrieval function
        Returns list of (object_index, similarity_score) tuples
        """
        
        log.info(f"Query: {query_path} ({query_modality})")
        log.info(f"Scene: {scan_id}")
        log.info(f"Target: {target_modality}")
        
        # Encode query
        query_embed = self.encode_query(query_path, query_modality)
        log.info(f"Query embedding shape: {query_embed.shape}")
        
        # Encode scene
        scene_data = self.encode_scene(scan_id)
        
        if target_modality not in scene_data:
            raise ValueError(f"Target modality {target_modality} not available in scene {scan_id}")
        
        target_embeddings = scene_data[target_modality]['embeddings']
        target_masks = scene_data[target_modality]['masks']
        
        # Filter valid objects only 
        valid_mask = target_masks.bool()
        if valid_mask.sum() == 0:
            log.warning("No valid objects found in target modality")
            return []
        
        valid_embeddings = target_embeddings[valid_mask]
        valid_indices = torch.where(valid_mask)[0]
        
        # Compute similarities
        sim = torch.softmax(query_embed.unsqueeze(0) @ valid_embeddings.t(), dim=-1)
        rank_list = torch.argsort(1.0 - sim, dim=1)
        top_k_indices = rank_list[0, :top_k]
        
        results = []
        object_ids = scene_data[target_modality]['object_ids']
        
        for i in range(min(top_k, len(top_k_indices))):
            valid_idx = top_k_indices[i].item()
            original_idx = valid_indices[valid_idx].item()
            similarity = sim[0, valid_idx].item()  # Get similarity from softmax result
            actual_object_id = object_ids[original_idx] if original_idx < len(object_ids) else original_idx
            results.append((actual_object_id, similarity))
        
        log.info(f"Found {len(results)} matches:")
        for i, (obj_id, sim) in enumerate(results):
            log.info(f"  {i+1}. Object ID {obj_id}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Cross Modal Instance Retrieval within a Scene',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Query arguments with defaults from config
    parser.add_argument('--query_path', type=str, default=DEFAULT_CONFIG['query_path'],
                       help=f'Path to query (point cloud, image, or text) - default: {DEFAULT_CONFIG["query_path"]}')
    parser.add_argument('--query_modality', type=str, default=DEFAULT_CONFIG['query_modality'],
                       choices=['point', 'rgb', 'referral'],
                       help=f'Query modality - default: {DEFAULT_CONFIG["query_modality"]}')
    parser.add_argument('--scan_id', type=str, default=DEFAULT_CONFIG['scan_id'],
                       help=f'Scene ID to search in - default: {DEFAULT_CONFIG["scan_id"]}')
    parser.add_argument('--target_modality', type=str, default=DEFAULT_CONFIG['target_modality'],
                       choices=['point', 'rgb', 'referral', 'cad'],
                       help=f'Target modality to match against - default: {DEFAULT_CONFIG["target_modality"]}')
    
    # Dataset arguments with defaults from config
    parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                       choices=['scannet', 'scan3r', 'arkitscenes', 'multiscan'],
                       help=f'Dataset name - default: {DEFAULT_CONFIG["dataset"]}')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_CONFIG['data_dir'],
                       help=f'Path to dataset directory - default: {DEFAULT_CONFIG["data_dir"]}')
    parser.add_argument('--process_dir', type=str, default=DEFAULT_CONFIG['process_dir'],
                       help=f'Path to preprocessed features directory - default: {DEFAULT_CONFIG["process_dir"]}')
    parser.add_argument('--ckpt', type=str, default=DEFAULT_CONFIG['ckpt'],
                       help=f'Path to model checkpoint - default: {DEFAULT_CONFIG["ckpt"]}')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=DEFAULT_CONFIG['top_k'],
                       help=f'Number of top results to return - default: {DEFAULT_CONFIG["top_k"]}')
    
    # Model dimensions
    parser.add_argument('--input_dim_3d', type=int, default=384)
    parser.add_argument('--input_dim_2d', type=int, default=1536)
    parser.add_argument('--input_dim_1d', type=int, default=768)
    parser.add_argument('--out_dim', type=int, default=768)
    
    args = parser.parse_args()
    
    # Print configuration being used
    log.info("=== Instance Retrieval Configuration ===")
    log.info(f"Dataset: {args.dataset}")
    log.info(f"Data directory: {args.data_dir}")
    log.info(f"Process directory: {args.process_dir}")
    log.info(f"Model checkpoint: {args.ckpt}")
    log.info(f"Query: {args.query_path} ({args.query_modality})")
    log.info(f"Scene: {args.scan_id}")
    log.info(f"Target modality: {args.target_modality}")
    log.info(f"Top-K: {args.top_k}")
    log.info("========================================")
    
    # Basic validation
    if not osp.exists(args.query_path) and args.query_modality != 'referral':
        log.warning(f"Query path does not exist: {args.query_path}")
    if not osp.exists(args.data_dir):
        log.warning(f"Data directory does not exist: {args.data_dir}")
    if not osp.exists(args.ckpt):
        log.warning(f"Checkpoint does not exist: {args.ckpt}")
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Run retrieval
    retriever = InstanceRetrieval(args)
    results = retriever.retrieve(
        args.query_path,
        args.query_modality, 
        args.scan_id,
        args.target_modality,
        args.top_k
    )
    
    return results


if __name__ == '__main__':
    main()
    