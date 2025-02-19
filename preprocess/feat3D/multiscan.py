import os.path as osp
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm

from common import load_utils 
from util import point_cloud, multiscan
from util.multiscan import MULTISCAN_SCANNET
from preprocess.build import PROCESSOR_REGISTRY
from preprocess.feat3D.base import Base3DProcessor

@PROCESSOR_REGISTRY.register()
class MultiScan3DProcessor(Base3DProcessor):
    def __init__(self, config_data, config_3D, split) -> None:
        super(MultiScan3DProcessor, self).__init__(config_data, config_3D, split)
        self.data_dir = config_data.base_dir
        
        files_dir = osp.join(config_data.base_dir, 'files')
        
        self.scan_ids = []
        self.scan_ids = multiscan.get_scan_ids(files_dir, split)
        
        self.out_dir = osp.join(config_data.process_dir, 'scans')
        load_utils.ensure_dir(self.out_dir)
        self.label_map = multiscan.read_label_map(files_dir, label_from = 'raw_category', label_to = 'nyu40id')
        
        self.undefined = 0        

        
    def load_objects_for_scan(self, scan_id):
        """Load and parse the annotations JSON for the given scan ID."""
        objects_path = osp.join(self.data_dir, 'scenes', scan_id, f"{scan_id}.annotations.json")
        if not osp.exists(objects_path):
            raise FileNotFoundError(f"Annotations file not found for scan ID: {scan_id}")
        
        annotations = load_utils.load_json(objects_path)
        
        objects = []
        
        for obj in annotations["objects"]:
            object_id=obj["objectId"]
            objectName=obj["label"].split('.')[0]
            scannet_class=MULTISCAN_SCANNET[objectName]
            nyu40id=self.label_map[scannet_class]
            objects.append({
                "objectId": object_id,
                "global_id": nyu40id
            })
        
        return objects


    
    def compute3DFeaturesEachScan(self, scan_id):
        ply_data = multiscan.load_ply_data(osp.join(self.data_dir, 'scenes'), scan_id)
        mesh_points = np.stack([ply_data['x'], ply_data['y'], ply_data['z']]).transpose((1, 0))
                
        mesh = o3d.io.read_triangle_mesh(osp.join(self.data_dir, 'scenes', scan_id,'{}.ply'.format(scan_id)))
        mesh_colors = np.asarray(mesh.vertex_colors)*255.0
        mesh_colors = mesh_colors.round()
                
        scan_objects=self.load_objects_for_scan(scan_id)
        
        object_pcl_embeddings, object_cad_embeddings = {}, {}
        object_id_to_label_id = {}
        for idx, scan_object in enumerate(scan_objects):
            instance_id = int(scan_object['objectId'])
            global_object_id = scan_object['global_id']

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
            
        torch.save(data3D, osp.join(scene_out_dir, 'data3D.pt'))
        torch.save(object_id_to_label_id_map, osp.join(scene_out_dir, 'object_id_to_label_id_map.pt'))
    