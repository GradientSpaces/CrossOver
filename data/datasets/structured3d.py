import os.path as osp
import numpy as np
from typing import List, Any
from omegaconf import DictConfig

from ..build import DATASET_REGISTRY
from .scanbase import ScanObjectBase, ScanBase

@DATASET_REGISTRY.register()
class Structured3DObject(ScanObjectBase):
    """Structured3D dataset class for instance level baseline"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__(data_config, split)

@DATASET_REGISTRY.register()
class Structured3D(ScanBase):
    """Structured3D dataset class"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__(data_config, split)
        
        filepath = osp.join(self.files_dir, '{}_scans.txt'.format(self.split))
        self.scan_ids = np.genfromtxt(filepath, dtype = str)
                   