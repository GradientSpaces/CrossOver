import os.path as osp
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
import os
import imageio
import skimage.transform as sktf
from PIL import Image
from scipy.spatial.transform import Rotation as R
from omegaconf import DictConfig
from typing import List, Dict, Tuple

from common import load_utils
from util import render, scannet, visualisation
from util import image as image_util

from preprocess.build import PROCESSOR_REGISTRY
from preprocess.feat2D.base import Base2DProcessor

@PROCESSOR_REGISTRY.register()
class Scannet2DProcessor(Base2DProcessor):
    """Scannet 2D (RGB + Floorplan) feature processor class."""
    def __init__(self, config_data: DictConfig, config_2D: DictConfig, split: str) -> None:
        super(Scannet2DProcessor, self).__init__(config_data, config_2D, split)
        self.split = split
        self.data_dir = config_data.base_dir
        files_dir = osp.join(config_data.base_dir, 'files')
        
        self.scan_ids = []
        self.scan_ids = scannet.get_scan_ids(files_dir, self.split)
        
        self.out_dir = osp.join(config_data.process_dir, 'scans')
        load_utils.ensure_dir(self.out_dir)
        
        self.orig_image_size = config_2D.image.orig_size
        self.model_image_size = config_2D.image.model_size
         
        self.frame_skip = config_data.skip_frames
        self.top_k = config_2D.image.top_k
        self.num_levels = config_2D.image.num_levels
        self.undefined = 0
        
        self.layout_dir = config_data.layout_dir
        self.shape_dir = config_data.shape_dir
        self.shape_annot = load_utils.load_json(osp.join(files_dir, 'scan2cad_full_annotations.json')) 
        
        self.frame_pose_data = {}
        for scan_id in self.scan_ids:
            pose_data = scannet.load_poses(osp.join(self.data_dir, 'scans'), scan_id, skip=self.frame_skip)
            self.frame_pose_data[scan_id] = pose_data
        
    def compute2DFeatures(self):
        for scan_id in tqdm(self.scan_ids): 
            self.compute2DFeaturesEachScan(scan_id)   
    
    def renderShapeAndFloorplan(self, scene_folder: str, scene_out_folder: str, scan_id: str) -> Image.Image:
        meta_file = osp.join(scene_folder, scan_id + '.txt')
        shape_annot = [cad_annot for cad_annot in self.shape_annot if cad_annot['id_scan'] == scan_id]
        scan_layout_filename = osp.join(self.layout_dir, f'{scan_id}.json')
        
        lines = open(meta_file).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
            
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))    
        
        scan_layout_and_object_mesh = scannet.makeShapeAndLayoutMesh(self.shape_dir, scan_layout_filename, axis_align_matrix, shape_annot)
        
        if scan_layout_and_object_mesh is None: 
            return None
        
        mesh_outpath = osp.join(scene_out_folder, 'floor+obj.ply')
        o3d.io.write_triangle_mesh(mesh_outpath, scan_layout_and_object_mesh)
        render_img = render.render_scene(mesh_outpath)
        
        # Crop to have tight bbox
        render_img = image_util.crop_image(render_img, mesh_outpath.replace('floor+obj.ply', 'floor+obj.png'))
        return render_img
         
    def compute2DFeaturesEachScan(self, scan_id: str) -> None:
        data2D = {}
        frame_idxs = list(self.frame_pose_data[scan_id].keys())
        scene_folder = osp.join(self.data_dir, 'scans', scan_id)
        
        scene_out_dir = osp.join(self.out_dir, scan_id)
        load_utils.ensure_dir(scene_out_dir)
        pt_2d_path = osp.join(scene_out_dir, 'data2D.pt')
        if osp.exists(pt_2d_path):
            print("using 2d pt")
            pt_data=torch.load(pt_2d_path)
            data2D['objects']=pt_data['objects']
            data2D['scene']=pt_data['scene']
            os.remove(pt_2d_path)
            
        else:
        # Floor-plan rendering
            render_img = self.renderShapeAndFloorplan(scene_folder, scene_out_dir, scan_id)
            floorplan_embeddings = None
            
            if render_img is not None:
                render_img = render_img.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
                render_img_pt = self.model.base_tf(render_img)
                floorplan_embeddings = self.extractFeatures([render_img_pt], return_only_cls_mean = False)            
            floorplan_dict = {'img' : render_img, 'embedding' : floorplan_embeddings}
                
            # Multi-view Image -- Object (Embeddings)
            object_image_embeddings, object_image_votes_topK = self.computeImageFeaturesAllObjectsEachScan(scene_folder, frame_idxs)
        
            # Multi-view Image -- Scene (Images + Embeddings)
            color_path = osp.join(scene_folder, 'data/color')
            intrinsic_data = scannet.load_intrinsics(osp.join(self.data_dir, 'scans'), scan_id)
        
            pose_data, scene_images_pt, scene_image_embeddings, sampled_frame_idxs = self.computeImageFeaturesEachScan(scan_id, color_path, frame_idxs)
        
            # Visualise
            scene_mesh = o3d.io.read_triangle_mesh(osp.join(scene_folder, scan_id + '_vh_clean_2.ply'))
            intrinsics = { 'f' : intrinsic_data['intrinsic_mat'][0, 0], 'cx' : intrinsic_data['intrinsic_mat'][0, 2], 'cy' : intrinsic_data['intrinsic_mat'][1, 2], 
                            'w' : int(intrinsic_data['width']), 'h' : int(intrinsic_data['height'])}
            
            cams_visualised_on_mesh = visualisation.visualise_camera_on_mesh(scene_mesh, pose_data[sampled_frame_idxs], intrinsics, stride=1)
            
            image_path = osp.join(scene_out_dir, 'sel_cams_on_mesh.png')
            Image.fromarray((cams_visualised_on_mesh * 255).astype(np.uint8)).save(image_path)
            
            data2D['objects'] = {'image_embeddings': object_image_embeddings, 'topK_images_votes' : object_image_votes_topK}
            data2D['scene']   = {'scene_embeddings': scene_image_embeddings, 'images' : scene_images_pt, 
                                    'frame_idxs' : frame_idxs, 'sampled_cam_idxs' : sampled_frame_idxs}
            
            data2D['scene']['floorplan'] = floorplan_dict
        # torch.save(data2D, osp.join(scene_out_dir, 'data2D.pt'))
        np.savez_compressed(osp.join(scene_out_dir, 'data2D.npz'), **data2D)
    
    def computeImageFeaturesEachScan(self, scan_id: str, color_path: str, frame_idxs: List[int]) -> Tuple[np.ndarray, List[torch.tensor], np.ndarray, List[int]]:
        # Sample Camera Indexes Based on Rotation Matrix From Grid
        pose_data = []
        for frame_idx in frame_idxs:
            pose = self.frame_pose_data[scan_id][frame_idx]
            rot_quat = R.from_matrix(pose[:3, :3]).as_quat()
            trans = pose[:3, 3]
            pose_data.append([trans[0], trans[1], trans[2], rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]])
            
        pose_data = np.array(pose_data)
        
        sampled_frame_idxs = image_util.sample_camera_pos_on_grid(pose_data)
        
        # Extract Scene Image Features
        scene_images_pt = []
        for idx in sampled_frame_idxs:
            frame_index = frame_idxs[idx]
            
            image = Image.open(osp.join(color_path, '{}.jpg'.format(frame_index)))
            image = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
            image_pt = self.model.base_tf(image)
            scene_images_pt.append(image_pt)
        
        scene_image_embeddings = self.extractFeatures(scene_images_pt, return_only_cls_mean= False)
        
        return pose_data, scene_images_pt, scene_image_embeddings, sampled_frame_idxs
    
    def computeImageFeaturesAllObjectsEachScan(self, scene_folder: str, scene_out_dir: str, frame_idxs: List[str]) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, List[int]]]:
        instance_path = osp.join(scene_folder, 'data/instance-filt')
        object_image_votes, object_anno_2d = {}, {}
        for frame_idx in frame_idxs:
            label_file = osp.join(instance_path, str(frame_idx) + '.png')
            image = np.array(imageio.imread(label_file))
            image = sktf.resize(image, self.orig_image_size, order=0, preserve_range=True)
            
            object_anno_2d[frame_idx] = image
            frame_object_ids, counts = np.unique(image, return_counts=True)
                
            for idx in range(len(frame_object_ids)):
                object_id, count_id = frame_object_ids[idx], counts[idx]
                if object_id == self.undefined: 
                    continue
                if object_id not in object_image_votes: 
                    object_image_votes[object_id] = {}
                object_image_votes[object_id][frame_idx] = count_id
        
        object_image_votes_topK = {}
        for object_id in object_image_votes:
            object_image_votes_topK[object_id] = []
            obj_image_votes_f = object_image_votes[object_id]
            sorted_frame_idxs = sorted(obj_image_votes_f, key=obj_image_votes_f.get, reverse=True)
            if len(sorted_frame_idxs) > self.top_k:
                object_image_votes_topK[object_id] = sorted_frame_idxs[:self.top_k]
            else:
                object_image_votes_topK[object_id] = sorted_frame_idxs
        
        object_image_embeddings = {}
        for object_id in object_image_votes_topK:
            object_image_votes_topK_frames = object_image_votes_topK[object_id]
            object_image_embeddings[object_id] = {}
            for frame_idx in object_image_votes_topK_frames:
                image_path = osp.join(scene_folder, 'data/color', str(frame_idx) + '.jpg')
                color_img = Image.open(image_path)
                object_image_embeddings[object_id][frame_idx] = self.computeImageFeaturesEachObject(color_img, object_id, object_anno_2d[frame_idx])
        
        return object_image_embeddings, object_image_votes_topK
    
    def computeImageFeaturesEachObject(self, image: Image.Image, object_id: int, object_anno_2d: np.ndarray) -> np.ndarray:
        object_mask = object_anno_2d == object_id
        
        images_crops = []
        for level in range(self.num_levels):
            mask_tensor = torch.from_numpy(object_mask).float()
            x1, y1, x2, y2 = image_util.mask2box_multi_level(mask_tensor, level)
            cropped_img = image.crop((x1, y1, x2, y2))
            cropped_img = cropped_img.resize((self.model_image_size[1], self.model_image_size[1]), Image.BICUBIC)
            
            img_pt = self.model.base_tf(cropped_img)
            images_crops.append(img_pt)
        
        if(len(images_crops) > 0):
            mean_feats = self.extractFeatures(images_crops, return_only_cls_mean = True)
        return mean_feats