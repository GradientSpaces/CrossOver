import os.path as osp
import h5py
import torch
import numpy as np
import copy
import random
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from copy import deepcopy
from omegaconf import DictConfig
from typing import List, Dict, Any

from ..transforms import get_transform
from ..data_utils import pad_tensors

class ScanObjectBase(Dataset):
    """Base Dataset class for instance level training"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        self.split = split
        self.chunked_filepath = osp.join(data_config.chunked_dir, f'{self.split}_objects.h5')
        
        self.file_handler = h5py.File(self.chunked_filepath, 'r')
        self.scan_ids = list(self.file_handler.keys())
        self.index_to_scan_object = []
        
        self.modalities = data_config.avail_modalities
        
        # Preprocess to map indices to specific (scan, object_id)
        for scan_id in self.scan_ids:
            scan_group = self.file_handler[scan_id]
            object_ids = [int(key.split('_')[1]) for key in scan_group if key.startswith('object_')]
            for object_id in object_ids:
                self.index_to_scan_object.append((scan_id, object_id))

        random.shuffle(self.index_to_scan_object)
        
    def __len__(self) -> int:
        # The total number of objects across all scans
        return len(self.index_to_scan_object)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Get the scan_id and object_id corresponding to the global idx
        scan_id, object_id = self.index_to_scan_object[index]
        
        object_data = {}
        object_data['masks'] = {}
        object_data['inputs'] = {}
        
        # Open the HDF5 file and retrieve the data for the specific objec
        object_group = self.file_handler[scan_id][f'object_{object_id}']
        
        # Retrieve the label ID for the specific object
        label_id = object_group['label_id'][()]
        
        # Retrieve the input data for each modality
        inputs = {}
        for modality_name in self.modalities:
            inputs[modality_name] = object_group[f'inputs/{modality_name}']
        
        object_data['label_id']  = label_id
        object_data['object_id'] = object_id
        object_data['scan_id']   = scan_id
        
        # Convert to PyTorch tensors
        label_id = torch.tensor(label_id, dtype=torch.int64)
        for modality_name in inputs:
            inputs[modality_name] = torch.tensor(inputs[modality_name], dtype=torch.float32)
            mask = torch.all(inputs[modality_name] == 0)
            mask = ~mask
            object_data['masks'][modality_name] = mask
            object_data['inputs'][modality_name] = inputs[modality_name]
        
        return object_data
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_dict = {}
        
        batch_dict['inputs'] = {}
        batch_dict['masks']  = {}
        
        batch_dict['object_ids'] = torch.from_numpy(np.array([data_dict['object_id'] for data_dict in batch]).astype(np.int32))
        batch_dict['label_ids']  = torch.from_numpy(np.array([data_dict['label_id'] for data_dict in batch]).astype(np.int32))
        batch_dict['batch_size'] = batch_dict['label_ids'].shape[0]
        batch_dict['scan_id']    = [data_dict['scan_id'] for data_dict in batch]
        batch_dict['batch_size'] = batch_dict['label_ids'].shape[0]
        
        for modalityType in self.modalities:
            batch_dict['inputs'][modalityType] = torch.stack([data_dict['inputs'][modalityType] for data_dict in batch])
            batch_dict['masks'][modalityType]  = torch.stack([data_dict['masks'][modalityType] for data_dict in batch])
    
        return batch_dict

class ScanBase(Dataset):
    """Base Dataset class"""
    
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__()
        self.process_dir = data_config.process_dir
        self.split = split
        self.files_dir = osp.join(data_config.base_dir, 'files')
        
        self.max_obj_len = data_config.max_object_len
        self.modalities = data_config.avail_modalities        
        self.voxel_size = data_config.voxel_size
    
    def __len__(self) -> int:
        return len(self.scan_ids)
    
    def _pad_scene(self, objects_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Pad scene objects to maximum length and handle masks for different modalities."""
        objects_dict['object_masks'] = (torch.arange(self.max_obj_len) < objects_dict['num_objects'])
        
        for modalityType in objects_dict['inputs'].keys():
            inputs = objects_dict['inputs'][modalityType]
            modality_mask = copy.deepcopy(objects_dict['masks'][modalityType])
            objects_dict['masks'][modalityType] = copy.deepcopy(objects_dict['object_masks'])
            objects_dict['masks'][modalityType][:objects_dict['num_objects']] = modality_mask
            
            if modalityType in objects_dict['object_locs']:
                object_locs = objects_dict['object_locs'][modalityType]
                objects_dict['object_locs'][modalityType] = pad_tensors(object_locs, lens=self.max_obj_len, pad=0.0).float()
            objects_dict['inputs'][modalityType] = pad_tensors(inputs, lens=self.max_obj_len, pad=0.0).float()
        
        objects_dict['object_ids'] = pad_tensors(objects_dict['object_ids'].to(torch.int32), lens=self.max_obj_len, pad=-100).long()
        objects_dict['label_ids']  = pad_tensors(objects_dict['label_ids'].to(torch.int32), lens=self.max_obj_len, pad=-100).long()  
        
        return objects_dict
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        scan_id = self.scan_ids[index]
        
        scan_process_dir = osp.join(self.process_dir, 'scans', scan_id)
        
        # scan_objects_data = torch.load(osp.join(scan_process_dir, 'objectsDataMultimodal.pt'))
        scan_objects_data = np.load(osp.join(scan_process_dir, 'objectsDataMultimodal.npz'), allow_pickle=True)
        
        # scandata_1d = torch.load(osp.join(scan_process_dir, 'data1D.pt'))
        scandata_1d = np.load(osp.join(scan_process_dir, 'data1D.npz'), allow_pickle=True)
        # scandata_2d = torch.load(osp.join(scan_process_dir, 'data2D.pt'))
        scandata_2d = np.load(osp.join(scan_process_dir, 'data2D.npz'), allow_pickle=True)
        # scandata_3d = torch.load(osp.join(scan_process_dir, 'data3D.pt'))
        scandata_3d = np.load(osp.join(scan_process_dir, 'data3D.npz'), allow_pickle=True)
        
        # Point Cloud Data -- Scene
        points, feats, scene_label = scandata_3d['scene'].item()['pcl_coords'], scandata_3d['scene'].item()['pcl_feats'], scandata_3d['scene'].item()['scene_label']
        feats /= 255.
        feats -= 0.5
        
        if scene_label is None:
            scene_label = 'NA'
        
        if self.split == 'train':
            points, feats, _ = get_transform()(points, feats)
            points, feats = points.numpy(), feats.numpy()
        
        _, sel = ME.utils.sparse_quantize(points / self.voxel_size, return_index=True)
        coords, feats = points[sel], feats[sel]
        
        # Get coords, already zero centered during preprocessing
        coords = np.floor(coords / self.voxel_size)
        # coords-=coords.min(0)
        
        # Object Data
        scene_dict = {}
        scene_dict['objects'] = {'inputs': {}, 'masks' : {}, 'object_locs' : {}}
        scene_dict['masks'] = {}
        
        for modality_name in self.modalities:
            inputs = torch.from_numpy(scan_objects_data['inputs'][modality_name])
            
            if len(inputs.shape) > 2:
                mask = (inputs == 0).all(dim=2).all(dim=1) 
            else:
                mask = torch.all(inputs == 0, dim=1)
            scene_dict['objects']['inputs'][modality_name] = inputs
            
            mask = ~mask
            scene_dict['objects']['masks'][modality_name] = mask
            
            if modality_name in scan_objects_data['object_locs']:
                scene_dict['objects']['object_locs'][modality_name] = torch.from_numpy(scan_objects_data['object_locs'][modality_name])
            
        num_objects = len(scan_objects_data['object_id2idx'].keys())
        obj_id_to_label_id_map  = scan_objects_data['object_id_to_label_id_map']
        
        object_idx2id = {v : k for k, v in scan_objects_data['object_id2idx'].items()}
        object_ids = np.array([object_idx2id[object_idx] for object_idx in range(num_objects)])
        label_ids  = np.array([obj_id_to_label_id_map[object_id] for object_id in object_ids])
        
        scene_dict['scene_masks'] = {}
        
        rgb_embedding = torch.from_numpy(scandata_2d['scene'].item()['scene_embeddings'])
        rgb_embedding = torch.concatenate([rgb_embedding[:, 0, :], rgb_embedding[:, 1:, :].mean(dim=1)], dim=1)
        scene_dict['rgb_embedding'] = rgb_embedding
        
        scene_dict['scene_masks']['rgb'] = torch.Tensor([1.0])
        scene_dict['scene_masks']['point'] = torch.Tensor([1.0])
        scene_dict['scene_masks']['object'] = torch.Tensor([1.0])
        
        referral_mask = torch.Tensor([0.0])       
        referral_embedding = scandata_1d['scene'].item()['referral_embedding']
        
        if referral_embedding is not None:
            referral_embedding = torch.from_numpy(referral_embedding[0]['feat']).reshape(-1,)
            referral_mask = torch.Tensor([1.0])
        else:
            referral_embedding = torch.zeros((scene_dict['rgb_embedding'].shape[-1] // 4, ))
        
        floorplan_embedding = scandata_2d['scene'].item()['floorplan']['embedding']
        floorplan_mask = torch.Tensor([0.0])
        if floorplan_embedding is not None:
            floorplan_embedding = torch.from_numpy(floorplan_embedding[0, 0]).reshape(-1, )
            floorplan_mask = torch.Tensor([1.0])
        else:
            floorplan_embedding = torch.zeros((scene_dict['rgb_embedding'].shape[-1] // 2, ))
        
        scene_dict['referral_embedding'], scene_dict['scene_masks']['referral'] = referral_embedding, referral_mask
        scene_dict['floorplan_embedding'], scene_dict['scene_masks']['floorplan'] = floorplan_embedding, floorplan_mask
        scene_dict['objects']['object_ids'] = torch.from_numpy(object_ids.astype(np.int32))
        scene_dict['objects']['label_ids'] = torch.from_numpy(label_ids.astype(np.int32))
        scene_dict['objects']['num_objects'] = object_ids.shape[0]
        
        scene_dict['scan_id'] = scan_id
        scene_dict['scene_label'] = scene_label
        scene_dict['objects'] = self._pad_scene(scene_dict['objects'])
        scene_dict['pcl_coords'] = coords
        scene_dict['pcl_feats']  = feats
        
        scene_dict['masks'] = deepcopy(scene_dict['objects']['masks'])
        scene_dict['label_ids'] = deepcopy(scene_dict['objects']['label_ids'])
        
        del scene_dict['objects']['masks']
        del scene_dict['objects']['label_ids']
        
        for key in scene_dict:
            assert scene_dict[key] is not None, key
        
        return scene_dict

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        coords, feats, new_batch = [],  [], []
        for sample in batch:
            coords.append(torch.from_numpy(sample['pcl_coords']).int())
            feats.append(torch.from_numpy(sample['pcl_feats']))
            
            del sample['pcl_coords']
            del sample['pcl_feats']
            
            # Hack to add temporal instance matching mask
            if 'label_ids_rio_subset_mask' not in sample:
                sample['label_ids_rio_subset_mask'] = torch.from_numpy(np.zeros((sample['label_ids'].shape[0], )))
            
            new_batch.append(sample)
        
        coordinates, features = ME.utils.sparse_collate(coords, feats)
                    
        # Now combine these new_batch dictionaries into a new dictionary
        collated_batch = {
            key: torch.utils.data._utils.collate.default_collate([d[key] for d in new_batch])
            for key in new_batch[0]
        }
        collated_batch['coordinates'], collated_batch['features'] = coordinates, features
        
        return collated_batch
