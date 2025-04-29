import os.path as osp
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
import json
from common import load_utils 
from util import structured3d
from preprocess.build import PROCESSOR_REGISTRY
from preprocess.feat3D.base import Base3DProcessor

@PROCESSOR_REGISTRY.register()
class Structured3D_3DProcessor(Base3DProcessor):
    def __init__(self, config_data, config_3D, split) -> None:
        super(Structured3D_3DProcessor, self).__init__(config_data, config_3D, split)
        self.data_dir = config_data.base_dir
        
        files_dir = osp.join(config_data.base_dir, 'files')
        
        self.scan_ids = []
        self.scan_ids = structured3d.get_scan_ids(files_dir, split)
        
        self.out_dir = config_data.process_dir
        load_utils.ensure_dir(self.out_dir)
        # self.undefined = 0      

    def compute3DFeaturesEachScan(self, scan_id):
        scan_id = scan_id.split('_')
        room_id = scan_id[-1]
        scan_id = scan_id[0]+'_'+scan_id[1]
        ply_data = structured3d.load_ply_data(osp.join(self.data_dir, 'scans'), scan_id, room_id)
        mesh_points = np.stack([ply_data['x'], ply_data['y'], ply_data['z']]).transpose((1, 0))
            
        # mesh = o3d.io.read_triangle_mesh(osp.join(self.data_dir, 'scans', scan_id, '3D_rendering', room_id, 'room_mesh.ply'))
        # mesh_colors = np.asarray(mesh.vertex_colors)*255.0
        mesh_colors = np.stack([ply_data['red'], ply_data['green'], ply_data['blue']]).transpose((1, 0))
        # print(mesh_colors)
        
        # mesh_colors = mesh_colors.round()
        object_ids = ply_data['objectId']
        unique_objects = np.unique(object_ids)
        # print(unique_objects)
        semantic_ids = ply_data['nyu40id']
        
        scene_label = None
        with open(osp.join(self.data_dir, 'scans', scan_id, 'annotation_3d.json')) as file:
            annotations = json.load(file)
        
        for annos in annotations['semantics']:
                if annos['ID'] == int(room_id):
                    scene_label = annos['type'].strip()
                    break


        object_pcl_embeddings, object_cad_embeddings = {}, {}
        object_id_to_label_id = {}
        
        for idx, instance_id in enumerate(unique_objects):
            object_pcl=mesh_points[np.where(ply_data['objectId'] == instance_id)]
            if object_pcl.shape[0] <= self.config_3D.min_points_per_object:
                continue
            
            assert instance_id not in object_id_to_label_id
            # first_point_idx = np.where(object_ids == instance_id)[0][0]
            # nyu40id = semantic_ids[first_point_idx]
            # object_id_to_label_id[instance_id] = nyu40id
            # Find the most common nyu40id for this object
            all_point_indices = np.where(object_ids == instance_id)[0]
            nyu40ids_for_object = semantic_ids[all_point_indices]
            unique_ids, counts = np.unique(nyu40ids_for_object, return_counts=True)
            nyu40id = unique_ids[np.argmax(counts)]
            object_id_to_label_id[instance_id] = nyu40id
            # if instance_id==0:
            #     print(nyu40id)
            
            if object_pcl.shape[0] >= self.config_3D.min_points_per_object:
                object_pcl_embeddings[instance_id] = self.normalizeObjectPCLAndExtractFeats(object_pcl)
            else:
                print("Object {} has less than {} points".format(instance_id, self.config_3D.min_points_per_object))
            
        # print(scene_label)
        data3D = {}    
        data3D['objects'] = {'pcl_embeddings' : object_pcl_embeddings, 'cad_embeddings': object_cad_embeddings}
        data3D['scene']   = {'pcl_coords': mesh_points, 'pcl_feats': mesh_colors, 'scene_label' : scene_label}
        # print(object_id_to_label_id)
        object_id_to_label_id_map = { 'obj_id_to_label_id_map' : object_id_to_label_id}
        
        assert len(list(object_id_to_label_id.keys())) >= len(list(object_pcl_embeddings.keys())), 'PC does not match for {}'.format(scan_id)
        scene_out_dir = osp.join(self.out_dir, scan_id+'_'+room_id)
        load_utils.ensure_dir(scene_out_dir)
            
        # torch.save(data3D, osp.join(scene_out_dir, 'data3D.pt'))
        # torch.save(object_id_to_label_id_map, osp.join(scene_out_dir, 'object_id_to_label_id_map.pt'))
        np.savez_compressed(osp.join(scene_out_dir, 'data3D.npz'), **data3D)
        np.savez_compressed(osp.join(scene_out_dir, 'object_id_to_label_id_map.npz'), **object_id_to_label_id_map)
    
