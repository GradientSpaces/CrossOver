import os.path as osp
import numpy as np
from functools import reduce
from operator import add
import torch
from tqdm import tqdm
from omegaconf import DictConfig
import h5py
from common import load_utils 
from common.constants import ModalityType
from util import scan3r, scannet, arkit, multiscan
from typing import Dict, Optional

from preprocess.build import PROCESSOR_REGISTRY

@PROCESSOR_REGISTRY.register()
class MultimodalPreprocessor:
    """
    A preprocessor class for handling multimodal data from 3D scans.
    Processes and combines data from different modalities (1D referral, 2D RGB/floorplan, 3D point cloud/CAD)
    and prepares it for training/evaluation.
    """

    def __init__(self, config_data: DictConfig, modality_config: DictConfig, split: str) -> None:
        self.split = split
        self.data_dir = config_data.base_dir
        self.dataset_name = self.data_dir.split('/')[-1]
        
        self.files_dir = osp.join(config_data.base_dir, 'files')
        self.scan_ids = []
        
        if self.dataset_name == 'Scannet':
            self.scan_ids = scannet.get_scan_ids(self.files_dir, self.split)
        elif self.dataset_name == 'Scan3R':
            self.scan_ids = scan3r.get_scan_ids(self.files_dir, self.split)
        elif self.dataset_name == 'ARKitScenes':
            self.scan_ids = arkit.get_scan_ids(self.files_dir, self.split)
        elif self.dataset_name == 'MultiScan':
            self.scan_ids = multiscan.get_scan_ids(self.files_dir, self.split)
        else:
            raise NotImplementedError
        
        self.filtered_scan_ids = []
        
        self.out_dir = osp.join(config_data.process_dir, 'scans')
        self.chunked_dir = config_data.chunked_dir
        
        load_utils.ensure_dir(self.chunked_dir)
        load_utils.ensure_dir(self.out_dir)
    
        self.undefined = 0
        self.feat_info = {}
        
        for modality in ModalityType.__dict__.values():
            if modality in ['referral']:
                self.feat_info[modality] = modality_config['1D'].feature_extractor.embed_dim
            if modality in ['rgb']:
                self.feat_info[modality] = modality_config['2D'].feature_extractor.embed_dim
            if modality in ['point', 'cad']:
                self.feat_info[modality] = modality_config['3D'].feature_extractor.embed_dim

        self.object_data = []
        
    def prepareData(self) -> None:
        """Prepare data for all scans in the dataset -- Creates an HDF5 file containing processed object features for each scan."""
        # Open HDF5 file to save datasets
        hdf5_filename = osp.join(self.chunked_dir, f'{self.split}_objects.h5')
        with h5py.File(hdf5_filename, 'w') as hf_handler:
            for scan_id in tqdm(self.scan_ids):
                self.prepareDataEachScan(scan_id, hf_handler) 
        
    def prepareObjectWiseDataEachScan(self, 
                                    out_dir: str, 
                                    data1D: Optional[Dict] = None, 
                                    data2D: Optional[Dict] = None, 
                                    data3D: Optional[Dict] = None) -> Dict:
        """Process object-wise data for a single scan combining features from all modalities."""
        object_id_to_label_id_map  = torch.load(osp.join(out_dir, 'object_id_to_label_id_map.pt'))['obj_id_to_label_id_map'] 
        map_object_ids = list(object_id_to_label_id_map.keys())
        
        precomputed_feats, inputs = {}, {}
        
        if data3D is not None:
            precomputed_feats[ModalityType.POINT] = data3D['objects']['pcl_embeddings'] 
            precomputed_feats[ModalityType.CAD] = data3D['objects']['cad_embeddings']
        if data2D is not None:
            precomputed_feats[ModalityType.RGB] = data2D['objects']['image_embeddings']
        if data1D is not None:
            precomputed_feats[ModalityType.REF] = data1D['objects']['referral_embeddings']
        
        object_ids = []
        for modalityType in ModalityType.__dict__.values():
            inputs[modalityType] = [] 
            object_ids.append(list(precomputed_feats[modalityType].keys()))
        
        object_ids = list(set(reduce(add, object_ids)))
        assert len(object_ids) <= len(map_object_ids), "No.of objects > mapped!"      
        
        for object_id in map_object_ids:
            if object_id not in object_ids: 
                del object_id_to_label_id_map[object_id]
                
        assert len(object_ids) == len(list(object_id_to_label_id_map.keys())), "Mapped != No.of objects!"
        
        object_locs = { ModalityType.POINT: [], ModalityType.CAD : []}
        
        for modalityType in self.feat_info:
            for object_id in object_ids:
                if object_id in precomputed_feats[modalityType]:
                    if modalityType == ModalityType.RGB:
                        feats = np.mean(list(precomputed_feats[modalityType][object_id].values()), axis=0).reshape(1, -1)
                    else:
                        feats = precomputed_feats[modalityType][object_id]['feats'].reshape(1, -1)

                    if modalityType in [ModalityType.POINT, ModalityType.CAD]:
                        object_loc = precomputed_feats[modalityType][object_id]['loc']
                    
                else:
                    feats = np.zeros((1, self.feat_info[modalityType]))
                    if modalityType in [ModalityType.POINT, ModalityType.CAD]:
                        object_loc = np.zeros((6,))
                if modalityType in [ModalityType.POINT, ModalityType.CAD]:
                    object_locs[modalityType].append(object_loc) 
                inputs[modalityType].append(feats) 
            
            inputs[modalityType] = np.concatenate(inputs[modalityType], axis=0)
            if modalityType in [ModalityType.POINT, ModalityType.CAD]:
                object_locs[modalityType] = np.array(object_locs[modalityType])
        
        # convert object id to the index in the tensor
        object_id2idx = {} 
        for index, v in enumerate(object_ids): 
            object_id2idx[v] = index
        
        for modalityType in self.feat_info:
            assert inputs[modalityType].shape[0] == len(object_id2idx), "Mapping does not match for {}!".format(modalityType)
        
        objects_data_pt = {
            'inputs': inputs,
            'object_locs' : object_locs,
            'object_id2idx' : object_id2idx,
            'object_id_to_label_id_map' : object_id_to_label_id_map,
            'object_ids' : object_ids,
            'topK_images_votes' : data2D['objects']['topK_images_votes']
        }
        
        torch.save(objects_data_pt, osp.join(out_dir, 'objectsDataMultimodal.pt'))
        return objects_data_pt
        
    def prepareDataEachScan(self, scan_id: str, hf_handler: h5py.File) -> None:
        """Process data for a single scan and store it in the HDF5 file."""
        out_dir = osp.join(self.out_dir, scan_id)
        
        data1D = torch.load(osp.join(out_dir, 'data1D.pt'))
        data2D = torch.load(osp.join(out_dir, 'data2D.pt'))
        data3D = torch.load(osp.join(out_dir, 'data3D.pt'))
        
        objects_data_pt = self.prepareObjectWiseDataEachScan(out_dir, data1D, data2D, data3D)
        self.dumpEachObjectDataPerScan(scan_id, objects_data_pt, hf_handler)
    
    def dumpEachObjectDataPerScan(self, 
                                 scan_id: str, 
                                 objects_data_pt: Dict, 
                                 hf_handler: h5py.File) -> None:
        """Store processed object features for a scan in the HDF5 file."""
        object_id_to_label_id_map = objects_data_pt['object_id_to_label_id_map']
        object_id2idx = objects_data_pt['object_id2idx']
        object_idx2id = {v : k for k, v in object_id2idx.items()}
        num_objects = len(objects_data_pt['object_id2idx'].keys())
        object_ids = np.array([object_idx2id[object_idx] for object_idx in range(num_objects)])
        label_ids  = np.array([object_id_to_label_id_map[object_id] for object_id in object_ids])
        inputs = objects_data_pt['inputs']
        
        # Create a group for the scan in the HDF5 file
        scan_group = hf_handler.create_group(scan_id)
        for idx, object_id in enumerate(object_ids):
            object_group = scan_group.create_group(f'object_{object_id}')
            object_group.create_dataset('label_id', data=np.int32(label_ids[idx]))
            
            for modalityType in inputs:
                object_group.create_dataset(
                    f'inputs/{modalityType}', 
                    data=np.array(inputs[modalityType][idx]).astype('float64'), 
                    compression="gzip", 
                    compression_opts=9)
        
    def run(self) -> None:
        """Execute the complete preprocessing pipeline."""
        self.prepareData()
