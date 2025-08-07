import os.path as osp
import numpy as np
from typing import List, Any
from omegaconf import DictConfig
import pandas as pd
from ..build import DATASET_REGISTRY
from .scanbase import ScanObjectBase, ScanBase

@DATASET_REGISTRY.register()
class ARKitScenesObject(ScanObjectBase):
    """ARKitScenes dataset class for instance level baseline"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__(data_config, split)

@DATASET_REGISTRY.register()
class ARKitScenes(ScanBase):
    """ARKitScenes dataset class"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__(data_config, split)
        
        filepath = osp.join(self.files_dir, '{}_scans.txt'.format(self.split))
        self.scan_ids = np.genfromtxt(filepath, dtype = str)
    
    def get_temporal_scan_pairs(self):
        """Groups scans into temporal pairs based on shared visit_id."""
        csv_path=osp.join(self.files_dir,'3dod_train_val_splits.csv')
        df = pd.read_csv(csv_path)

        df = df[df["visit_id"].notna()]

        grouped_scans = df.groupby("visit_id")["video_id"].apply(list).to_dict()

        scene_pairs = []
        for video_ids in grouped_scans.values():
            if len(video_ids) > 1: 
                ref_scan_id = video_ids[0]  # First video_id as reference
                rescan_list = [{"scan_id": rescan_id} for rescan_id in video_ids[1:]] 
                
                scene_pairs.append([ref_scan_id, rescan_list])
        
        return scene_pairs