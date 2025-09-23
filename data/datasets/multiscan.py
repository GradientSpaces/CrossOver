import os.path as osp
import numpy as np
from typing import List, Any
from omegaconf import DictConfig

from ..build import DATASET_REGISTRY
from .scanbase import ScanObjectBase, ScanBase

@DATASET_REGISTRY.register()
class MultiScanObject(ScanObjectBase):
    """MultiScan dataset class for instance level baseline"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__(data_config, split)

@DATASET_REGISTRY.register()
class MultiScan(ScanBase):
    """MultiScan dataset class"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__(data_config, split)
        
        filepath = osp.join(self.files_dir, '{}_scans.txt'.format(self.split))
        self.scan_ids = np.genfromtxt(filepath, dtype = str)
    
    def get_temporal_scan_pairs(self) -> List[List[Any]]:
        """Gets pairs of temporal scans from the dataset."""
        scene_pairs = []
        
        ref_scan_ids = [scan_id for scan_id in self.scan_ids if scan_id.endswith('00')]
        
        for ref_scan_id in ref_scan_ids:    
            rescan_list = []
            
            for rescan_id in self.scan_ids:
                rescan = {}
                if rescan_id.startswith(ref_scan_id.split('_')[0]) and rescan_id != ref_scan_id:
                    rescan['scan_id'] = rescan_id
                    rescan_list.append(rescan)
            if len(rescan_list) == 0: 
                continue
            
            scene_pairs.append([ref_scan_id, rescan_list])
        return scene_pairs