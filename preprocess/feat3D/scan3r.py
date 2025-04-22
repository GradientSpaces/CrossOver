import os.path as osp
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from typing import Any, Dict

from common import load_utils 
from util import point_cloud, scan3r

from preprocess.build import PROCESSOR_REGISTRY
from preprocess.feat3D.base import Base3DProcessor

@PROCESSOR_REGISTRY.register()
class Scan3R3DProcessor(Base3DProcessor):
    """Scan3R 3D feature (point cloud) processor class."""
    def __init__(self, config_data: DictConfig, config_3D: DictConfig, split: str) -> None:
        super(Scan3R3DProcessor, self).__init__(config_data, config_3D, split)
        self.data_dir = config_data.base_dir
        
        files_dir = osp.join(config_data.base_dir, 'files')
        
        self.scan_ids = []
        self.scan_ids = scan3r.get_scan_ids(files_dir, split)
        
        self.out_dir = osp.join(config_data.process_dir, 'scans')
        load_utils.ensure_dir(self.out_dir)
        
        self.label_filename = config_data.label_filename
        self.undefined = 0
        
        self.objects = load_utils.load_json(osp.join(files_dir, 'objects.json'))['scans']
        
        # get device 
        if not torch.cuda.is_available(): 
            raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")
        
        self.config_3D = config_3D
        
        # load feature extractor
        self.feature_extractor = self.loadFeatureExtractor(config_3D, "3D")
    
    def compute3DFeaturesEachScan(self, scan_id: str) -> None:
        """
        Computes 3D features for a single scan.
        """
        ply_data = scan3r.load_ply_data(osp.join(self.data_dir, 'scans'), scan_id, self.label_filename)
        mesh_points = np.stack([ply_data['x'], ply_data['y'], ply_data['z']]).transpose((1, 0))
        
        mesh = o3d.io.read_triangle_mesh(osp.join(self.data_dir, 'scans', scan_id, self.label_filename))
        mesh_colors = np.asarray(mesh.vertex_colors)*255.0
        mesh_colors = mesh_colors.round()
        
        scan_objects = [obj_data for obj_data in self.objects if obj_data['scan'] == scan_id][0]['objects']
        
        object_pcl_embeddings, object_cad_embeddings = {}, {}
        object_id_to_label_id = {}
        for idx, scan_object in enumerate(scan_objects):
            instance_id = int(scan_object['id'])
            global_object_id = int(scan_object['global_id'])

            object_pcl = mesh_points[np.where(ply_data['objectId'] == instance_id)]
            
            if object_pcl.shape[0] <= self.config_3D.min_points_per_object: 
                continue
            
            assert instance_id not in object_id_to_label_id
            object_id_to_label_id[instance_id] = global_object_id
            
            if object_pcl.shape[0] >= self.config_3D.min_points_per_object:
                object_pcl_embeddings[instance_id] = self.normalizeObjectPCLAndExtractFeats(object_pcl)

        data3D = {}    
        data3D['objects'] = {'pcl_embeddings' : object_pcl_embeddings, 'cad_embeddings': object_cad_embeddings}
        data3D['scene']   = {'pcl_coords': mesh_points[ply_data['objectId'] != self.undefined], 'pcl_feats': mesh_colors[ply_data['objectId'] != self.undefined], 'scene_label' : None}
            
        object_id_to_label_id_map = { 'obj_id_to_label_id_map' : object_id_to_label_id}
        
        assert len(list(object_id_to_label_id.keys())) >= len(list(object_pcl_embeddings.keys())), 'PC does not match for {}'.format(scan_id)
        scene_out_dir = osp.join(self.out_dir, scan_id)
        load_utils.ensure_dir(scene_out_dir)
            
        # torch.save(data3D, osp.join(scene_out_dir, 'data3D.pt'))
        # torch.save(object_id_to_label_id_map, osp.join(scene_out_dir, 'object_id_to_label_id_map.pt'))
        pt_data3d_path = osp.join(scene_out_dir, 'data3D.pt')
        pt_map_path = osp.join(scene_out_dir, 'object_id_to_label_id_map.pt')
        if osp.exists(pt_data3d_path):
            os.remove(pt_data3d_path)
        if osp.exists(pt_map_path): 
            os.remove(pt_map_path)
        np.savez_compressed(osp.join(scene_out_dir, 'data3D.npz'), **data3D)
        np.savez_compressed(osp.join(scene_out_dir, 'object_id_to_label_id_map.npz'), **object_id_to_label_id_map)