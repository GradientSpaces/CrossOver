import os.path as osp
import argparse
from accelerate import Accelerator
from datetime import timedelta
import torch
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm
import random
import numpy as np
import logging as log

import sys
sys.path.append(osp.abspath('.'))

from single_inference import datasets
from model.scene_crossover import SceneCrossOverModel
from util import torch_util

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

def run_inference(args, scan_id=None):
    if args.dataset == 'Scannet':
        dataset = datasets.ScannetInferDataset(args.data_dir, args.floorplan_dir)
    elif args.dataset == 'Scan3R':
        dataset = datasets.Scan3RInferDataset(args.data_dir)
    elif args.dataset == 'ARKitScenes':
        dataset = datasets.ARKitScenesInferDataset(args.data_dir)
    elif args.dataset == 'MultiScan':
        dataset = datasets.MultiScanInferDataset(args.data_dir)
    else:
        raise NotImplementedError('Dataset not implemented')
    
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    kwargs = [init_kwargs]
    accelerator = Accelerator(kwargs_handlers=kwargs)
    
    model = SceneCrossOverModel(args, accelerator.device)
    model = accelerator.prepare(model)
    model.eval()
    model.to(model.device)
    torch_util.load_weights(model, args.ckpt, accelerator.device)
    
    # # hack to calculate the number of parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # total_params += sum(p.numel() for p in model.encoder2D.dino_model.parameters())
    # total_params += sum(p.numel() for p in model.encoder1D.parameters())
    # print(f'Total number of parameters: {total_params}')
    # assert False
    
    data = { 'scene': []}
    if scan_id is not None:
        data_dict = dataset[scan_id]
        with torch.no_grad():
            output = model(data_dict)
        
        output_np = {}
        for modality in output['embeddings']:
            output_np[modality] = output['embeddings'][modality].cpu().numpy()
        torch.save(f'embed_{args.dataset.lower()}_{scan_id}.pt', {'scene': {'scan_id': scan_id, 'scene_embeds': output_np, 'masks': output['masks']}})
        
    else:
        for idx, scan_id in tqdm(enumerate(dataset.scan_ids)):
            data_dict = dataset[idx]
            with torch.no_grad():
                output = model(data_dict)
                
                output_np = {}
                for modality in output['embeddings']:
                    output_np[modality] = output['embeddings'][modality].cpu().numpy()
                
                data['scene'].append({'scan_id': scan_id, 'scene_embeds': output_np, 'masks': output['masks']})
            
        torch.save(data, f'/drive/dumps/multimodal-spaces/release_data/embed_{args.dataset.lower()}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene Inference')
    parser.add_argument('--dataset', default='Scannet', type=str, required=False)
    parser.add_argument('--data_dir', default='/drive/datasets/Scannet', type=str, required=False)
    parser.add_argument('--floorplan_dir', default='/drive/dumps/multimodal-spaces/preprocess_feats/Scannet/scans', type=str, required=False)
    parser.add_argument('--ckpt', default='/drive/dumps/multimodal-spaces/runs/UnifiedTrain_Scannet+Scan3R/2025-01-29-13:36:55.217923/ckpt/best.pth/', type=str, required=False)
    parser.add_argument('--scan_id', default='', type=str, required=False)
    parser.add_argument('--input_dim_3d', default=512, type=int, required=False)
    parser.add_argument('--input_dim_2d', default=1536, type=int, required=False)
    parser.add_argument('--input_dim_1d', default=768, type=int, required=False)
    parser.add_argument('--out_dim', default=768, type=int, required=False)
    
    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Parse the arguments
    args = parser.parse_args()
    run_inference(args, scan_id=None if args.scan_id == '' else args.scan_id)