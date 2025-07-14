"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
from abc import abstractmethod

from utils.config import ProcessUnit, EntryBase


class DatasetBase(EntryBase):
    """
    The base class of dataset
    """
    @abstractmethod
    def format_dataset(self, proc_unit: ProcessUnit):
        """
        Construct 3D point cloud from views
        """