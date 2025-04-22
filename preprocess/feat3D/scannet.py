import os.path as osp
import os
import numpy as np
import torch
from omegaconf import DictConfig

from common import load_utils 
from util import scannet, labelmap, render

from preprocess.build import PROCESSOR_REGISTRY
from preprocess.feat3D.base import Base3DProcessor

@PROCESSOR_REGISTRY.register()
class Scannet3DProcessor(Base3DProcessor):
    """Scannet 3D (Point Cloud + CAD) feature processor class."""
    def __init__(self, config_data: DictConfig, config_3D: DictConfig, split: str) -> None:
        super(Scannet3DProcessor, self).__init__(config_data, config_3D, split)
        self.data_dir = config_data.base_dir
        files_dir = osp.join(config_data.base_dir, 'files')
        
        self.scan_ids = []
        self.scan_ids = scannet.get_scan_ids(files_dir, split)
        
        self.layout_dir = config_data.layout_dir
        self.shape_dir = config_data.shape_dir
        self.shape_annot = load_utils.load_json(osp.join(files_dir, 'scan2cad_full_annotations.json')) 
        
        self.out_dir = osp.join(config_data.process_dir, 'scans')
        load_utils.ensure_dir(self.out_dir)
        
        self.load_files = {}
        for scan_id in self.scan_ids:
            scene_folder = osp.join(self.data_dir, 'scans', scan_id)
            mesh_file = osp.join(scene_folder, scan_id + config_data.mesh_subfix)
            aggre_file = osp.join(scene_folder, scan_id + config_data.aggre_subfix)
            seg_file = osp.join(scene_folder, scan_id + config_data.seg_subfix)
            meta_file = osp.join(scene_folder, scan_id + '.txt')
            
            self.load_files[scan_id] = {}
            self.load_files[scan_id]['mesh'] = mesh_file
            self.load_files[scan_id]['aggre'] = aggre_file
            self.load_files[scan_id]['seg']  = seg_file
            self.load_files[scan_id]['meta'] = meta_file
        
        # label map
        self.label_map = scannet.read_label_map(files_dir, label_from = 'raw_category', label_to = 'nyu40id')
        self.undefined = 0
        
        self.color_map = labelmap.get_NYU40_color_palette()
    
    def compute3DFeaturesEachScan(self, scan_id: str) -> None:
        mesh_file = self.load_files[scan_id]['mesh']
        aggre_file = self.load_files[scan_id]['aggre'] 
        seg_file = self.load_files[scan_id]['seg']
        meta_file = self.load_files[scan_id]['meta']
        
        scene_out_dir = osp.join(self.out_dir, scan_id)
        load_utils.ensure_dir(scene_out_dir)
        
        mesh_vertices, _, instance_ids, instance_bboxes, object_id_to_label_id, axis_align_matrix \
                                                                = scannet.export(mesh_file, aggre_file, seg_file, meta_file, self.label_map, 
                                                                                axis_alignment = True, output_file=None)
        
        mesh_points = mesh_vertices[:, 0:3] 
        mesh_colors  = mesh_vertices[:, 3:]
        
        center_points = np.mean(mesh_points, axis=0)
        center_points[2] = np.min(mesh_points[:, 2])
        mesh_points = mesh_points - center_points
        
        text_file = mesh_file.replace('_vh_clean_2.labels.ply' , '.txt')
        with open(text_file, 'r') as file:
                for line in file:
                    if line.startswith('sceneType'):
                        scene_label = line.split('=')[1].strip()  # Extract and clean the scene type
        
        unique_instance_ids = np.unique(instance_ids)
        object_pcl_embeddings, object_cad_embeddings = {}, {}
        
        shape_annot = [cad_annot for cad_annot in self.shape_annot if cad_annot['id_scan'] == scan_id]
        shape_annot_to_instance_map = None
         
        if len(shape_annot) > 0: 
            shape_annot = shape_annot[0]
            shape_annot_to_instance_map = scannet.get_cad_model_to_instance_mapping(instance_bboxes, shape_annot, meta_file, self.shape_dir)
        
        for instance_id in unique_instance_ids:
            if instance_id == self.undefined: 
                continue
            
            color = list(self.color_map[object_id_to_label_id[instance_id]])
            color = np.array(color).astype('float') / 255.0
            
            object_pcl = mesh_points[instance_ids == instance_id]
            if object_pcl.shape[0] >= self.config_3D.min_points_per_object:
                object_pcl_embeddings[instance_id] = self.normalizeObjectPCLAndExtractFeats(object_pcl)
            
            if shape_annot_to_instance_map is not None and instance_id in shape_annot_to_instance_map:
                shape_annot_instance = shape_annot_to_instance_map[instance_id]
                object_cad_pcl = shape_annot_instance['points']
                object_cad_embeddings[instance_id] = self.normalizeObjectPCLAndExtractFeats(object_cad_pcl)
            
        data3D = {}    
        data3D['objects'] = {'pcl_embeddings' : object_pcl_embeddings, 'cad_embeddings': object_cad_embeddings}
        data3D['scene']   = {'pcl_coords': mesh_points[instance_ids != self.undefined], 'pcl_feats': mesh_colors[instance_ids != self.undefined], 'scene_label' : scene_label}
        
        object_id_to_label_id_map = { 'obj_id_to_label_id_map' : object_id_to_label_id}
        
        assert len(list(object_id_to_label_id.keys())) >= len(list(object_pcl_embeddings.keys())), 'PC does not match for {}'.format(scan_id)
        assert len(list(object_id_to_label_id.keys())) >= len(list(object_cad_embeddings.keys())), 'CAD does not match for {}'.format(scan_id)
        
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