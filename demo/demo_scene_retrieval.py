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
from common.load_utils import load_yaml
import albumentations as A
import sys
sys.path.append(osp.abspath('.'))
from model.scene_crossover import SceneCrossOverModel
from util import torch_util

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

def load_data_and_get_embed(model, path, color_stats_path, query_modality, voxel_size=0.02, image_size=[224, 224]):
    base_tf = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])
    
    if query_modality == 'point':
        color_mean_std_path = osp.join(color_stats_path, 'color_mean_std.yaml')
        color_mean_std = load_yaml(color_mean_std_path)
        color_mean, color_std = (
            tuple(color_mean_std["mean"]),
            tuple(color_mean_std["std"]),
        )
        normalize_color = A.Normalize(mean=color_mean, std=color_std)
        assert path.endswith('.ply'), 'Point Cloud Path should be a .ply file!'
        pcd = o3d.io.read_point_cloud(path) 
        points = np.asarray(pcd.points)
        feats = np.asarray(pcd.colors)*255.0
        feats = feats.round()
        pseudo_image = feats.astype(np.uint8)[np.newaxis, :, :]
        feats = np.squeeze(normalize_color(image=pseudo_image)["image"])
        # feats -= 0.5
        
        coords, feats = torch_util.convert_to_sparse_tensor(points, feats, voxel_size)
        with torch.no_grad():
            embed = model.encode_3d(coords, feats).cpu()
    
    elif query_modality == 'rgb':
        assert os.isdir(path), 'RGB Path should be a directory'
        image_filenames = os.listdir(path)
        
        image_data = None
        for image_filename in image_filenames:
            image = Image.open(osp.join(path, image_filename))
            image = image.resize((image_size[1], image_size[0]), Image.BICUBIC)
            image_pt = base_tf(image).unsqueeze(0)
            image_data = image_pt if image_data is None else torch.cat((image_data, image_pt), dim=0)

        with torch.no_grad():
            embed = model.encode_rgb(image_data.to(model.device)).cpu()
    
    elif query_modality == 'floorplan':
        assert os.isfile(path), 'Floorplan Path should be a file'
        floorplan_img = Image.open(path)
        
        floorplan_img = floorplan_img.resize((image_size[1], image_size[0]), Image.BICUBIC)
        floorplan_data = base_tf(floorplan_img).unsqueeze(0)
        
        with torch.no_grad():
            embed = model.encode_floorplan(floorplan_data.to(model.device)).cpu()
    
    elif query_modality == 'referral':
        assert os.isfile(path), 'Referral Path should be a text file'
        with open(path, 'r') as f:
            text = f.read()
        
        text = [text]
        with torch.no_grad():
            embed = model.encode_1d(text).cpu()
    
    else:
        raise NotImplementedError('Query modality not implemented')

    return embed

def run_retrieval(query_embed, query_modality, database_embeds, database_modality):
    query_database_embeds = torch.from_numpy(np.array([embed_dict['scene_embeds'][database_modality].reshape(-1,) for embed_dict in database_embeds['scene']]))
    embed_masks = torch.tensor([embed_dict['masks'][database_modality]for embed_dict in database_embeds['scene']])
    
    query_database_embeds = query_database_embeds[embed_masks]
    sim = torch.softmax(query_embed @ query_database_embeds.t(), dim=-1)
    rank_list = torch.argsort(1.0 - sim, dim = 1)
    top_k_indices = rank_list[0]
    
    retrieved_scan_ids = [database_embeds['scene'][idx]['scan_id'] for idx in top_k_indices][:5]
    
    message = f'Query modality: {query_modality}, Database modality: {database_modality}'
    message += f"\n Top 5 retrieved scans: {retrieved_scan_ids}"
    log.info(message)

def run_inference(args):
    # load model
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    kwargs = [init_kwargs]
    accelerator = Accelerator(kwargs_handlers=kwargs)
    model = SceneCrossOverModel(args, accelerator.device)
    model = accelerator.prepare(model)
    model.eval()
    model.to(model.device)
    torch_util.load_weights(model, args.ckpt, accelerator.device)
    
    # load query data and get embed
    query_embed = load_data_and_get_embed(model, args.query_path, 'point')
    
    # load database data
    database_embeds = torch.load(args.database_path)
    log.info(f'Loaded database embeddings from {args.database_path}')
    
    run_retrieval(query_embed, args.query_modality, database_embeds, args.database_modality)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene Retrieval')
    parser.add_argument('--query_path', default='./demo_data/dining_room/scene_cropped.ply', type=str, required=False)
    parser.add_argument('--database_path', default='/drive/dumps/multimodal-spaces/release_data/embed_scannet.pt', type=str, required=False)
    parser.add_argument('--query_modality', default='point', type=str, required=False)
    parser.add_argument('--database_modality', default='referral', type=str, required=False)
    parser.add_argument('--database_stats_path', default='/drive/dumps/multimodal-spaces/preprocess_feats/Scannet', type=str, required=False)
    parser.add_argument('--ckpt', default='/drive/dumps/multimodal-spaces/runs/release_runs/scene_crossover_scannet+scan3r.pth/', type=str, required=False)
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
    run_inference(args)