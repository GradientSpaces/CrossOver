import os.path as osp
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R
from omegaconf import DictConfig
from typing import List, Dict, Tuple
import os
from common import load_utils
from util import render, scan3r, visualisation
from util import image as image_util


from preprocess.build import PROCESSOR_REGISTRY
from preprocess.feat2D.base import Base2DProcessor

@PROCESSOR_REGISTRY.register()
class Scan3R2DProcessor(Base2DProcessor):
    """Scan3R 2D (RGB) feature processor class."""
    def __init__(self, config_data: DictConfig, config_2D: DictConfig, split: str) -> None:
        super(Scan3R2DProcessor, self).__init__(config_data, config_2D, split)
        self.data_dir = config_data.base_dir
        files_dir = osp.join(config_data.base_dir, 'files')
        
        self.scan_ids = []
        self.split = split
        self.scan_ids = scan3r.get_scan_ids(files_dir, self.split)
        
        self.out_dir = osp.join(config_data.process_dir, 'scans')
        load_utils.ensure_dir(self.out_dir)
        
        self.orig_image_size = config_2D.image.orig_size
        self.model_image_size = config_2D.image.model_size
        
        self.frame_skip = config_data.skip_frames
        self.top_k = config_2D.image.top_k
        self.num_levels = config_2D.image.num_levels
        self.undefined = 0
        
        self.label_filename = config_data.label_filename
        
        # get frame_indexes
        self.frame_pose_data = {}
        for scan_id in self.scan_ids:
            scene_folder = osp.join(self.data_dir, 'scans', scan_id)
            frame_idxs = scan3r.load_frame_idxs(scene_folder)
            pose_data = scan3r.load_all_poses(scene_folder, frame_idxs)
            self.frame_pose_data[scan_id] = pose_data

    def compute2DFeatures(self) -> None:
        for scan_id in tqdm(self.scan_ids):
            self.compute2DImagesAndSeg(scan_id)
            self.compute2DFeaturesEachScan(scan_id)   

    def compute2DImagesAndSeg(self, scan_id: str) -> None:
            scene_folder = osp.join(self.data_dir, 'scans', scan_id)
            mesh_file = osp.join(scene_folder, self.label_filename.replace('.align', ''))
            obj_id_imgs = {}
            gt_pt_path = osp.join(scene_folder, 'gt-projection-seg.pt')
            if osp.exists(gt_pt_path):
                # print("using gt pt")
                old_gt = torch.load(gt_pt_path)
                for frame_idx in self.frame_pose_data[scan_id]:
                    obj_id_imgs[frame_idx] = old_gt[frame_idx]
                os.remove(gt_pt_path)
                    
            else:
                ply_data = scan3r.load_ply_data(self.data_dir, scene_folder, self.label_filename)
                instance_ids = ply_data['objectId']
                
                camera_info = scan3r.load_intrinsics(scene_folder)
                intrinsics = camera_info['intrinsic_mat']
                img_width = int(camera_info['width'])
                img_height = int(camera_info['height'])
                
                mesh = o3d.io.read_triangle_mesh(mesh_file)
                mesh_triangles = np.asarray(mesh.triangles)
                colors = np.asarray(mesh.vertex_colors)*255.0
                colors = colors.round()
                num_triangles = mesh_triangles.shape[0]
                
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
                
                # project 3D model
                for frame_idx in self.frame_pose_data[scan_id]:
                    img_pose = self.frame_pose_data[scan_id][frame_idx]
                    img_pose_inv = np.linalg.inv(img_pose)
                    
                    obj_id_map = render.project_mesh3DTo2D_with_objectseg(
                        scene, intrinsics, img_pose_inv, img_width, img_height, 
                        mesh_triangles, num_triangles, instance_ids
                    )
                    obj_id_imgs[frame_idx] = obj_id_map

            
            # save scene-level file for efficient loading
            scene_out_dir = osp.join(self.out_dir, scan_id)
            load_utils.ensure_dir(scene_out_dir)
            
            # torch.save(obj_id_imgs, osp.join(scene_out_dir, 'gt-projection-seg.pt'))
            np.savez_compressed(osp.join(scene_out_dir,'gt-projection-seg.npz'),**obj_id_imgs)
    
    def compute2DFeaturesEachScan(self, scan_id: str) -> None:
        scene_folder = osp.join(self.data_dir, 'scans', scan_id)
        color_path = osp.join(scene_folder, 'sequence')
        
        scene_out_dir = osp.join(self.out_dir, scan_id)
        load_utils.ensure_dir(scene_out_dir)
        
        obj_id_to_label_id_map = torch.load(osp.join(scene_out_dir, 'object_id_to_label_id_map.pt'))['obj_id_to_label_id_map']
        
        # Multi-view Image -- Object (Embeddings)
        object_image_embeddings, object_image_votes_topK, frame_idxs = self.computeImageFeaturesAllObjectsEachScan(scene_folder, scene_out_dir, obj_id_to_label_id_map)
        
        # Multi-view Image -- Scene (Images + Embeddings)
        frame_idxs = list(self.frame_pose_data[scan_id].keys())
        pose_data, scene_images_pt, scene_image_embeddings, sampled_frame_idxs = self.computeSelectedImageFeaturesEachScan(scan_id, color_path, frame_idxs)
        
        # Visualise
        camera_info = scan3r.load_intrinsics(scene_folder)
        intrinsic_mat = camera_info['intrinsic_mat']
        
        scene_mesh = o3d.io.read_triangle_mesh(osp.join(scene_folder, self.label_filename.replace('.align', '')))
        intrinsics = { 'f' : intrinsic_mat[0, 0], 'cx' : intrinsic_mat[0, 2], 'cy' : intrinsic_mat[1, 2], 
                        'w' : int(camera_info['width']), 'h' : int(camera_info['height'])}
        
        cams_visualised_on_mesh = visualisation.visualise_camera_on_mesh(scene_mesh, pose_data[sampled_frame_idxs], intrinsics, stride=1)
        image_path = osp.join(scene_out_dir, 'sel_cams_on_mesh.png')
        Image.fromarray((cams_visualised_on_mesh * 255).astype(np.uint8)).save(image_path)
        
        data2D = {}
        data2D['objects'] = {'image_embeddings': object_image_embeddings, 'topK_images_votes' : object_image_votes_topK}
        data2D['scene']   = {'scene_embeddings': scene_image_embeddings, 'images' : scene_images_pt, 
                                'frame_idxs' : frame_idxs, 'sampled_cam_idxs' : sampled_frame_idxs}
        
        # dummy floorplan
        floorplan_dict = {'img' : None, 'embedding' : None}
        data2D['scene']['floorplan'] = floorplan_dict
        
        torch.save(data2D, osp.join(scene_out_dir, 'data2D.pt'))
    
    def computeSelectedImageFeaturesEachScan(self, scan_id: str, color_path: str, frame_idxs: List[int]) -> Tuple[np.ndarray, List[torch.tensor], np.ndarray, List[int]]:
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
            
            image = Image.open(osp.join(color_path, f'frame-{frame_index}.color.jpg'))
            image = image.transpose(Image.ROTATE_270)
            image = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
            image_pt = self.model.base_tf(image)
            scene_images_pt.append(image_pt)
        
        scene_image_embeddings = self.extractFeatures(scene_images_pt, return_only_cls_mean= False)
        
        return pose_data, scene_images_pt, scene_image_embeddings, sampled_frame_idxs
    
    def computeImageFeaturesAllObjectsEachScan(self, scene_folder: str, scene_out_dir: str, obj_id_to_label_id_map: dict) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, List[int]], List[str]]:
        object_anno_2D = torch.load(osp.join(scene_out_dir, 'gt-projection-seg.pt'))
        object_image_votes = {}
        
        # iterate over all frames
        for frame_idx in object_anno_2D:
            obj_2D_anno_frame = object_anno_2D[frame_idx]
            # process 2D anno
            obj_ids, counts = np.unique(obj_2D_anno_frame, return_counts=True)
            for idx in range(len(obj_ids)):
                obj_id = obj_ids[idx]
                count = counts[idx]
                if obj_id == self.undefined:
                    continue
                
                if obj_id not in object_image_votes:
                    object_image_votes[obj_id] = {}
                if frame_idx not in object_image_votes[obj_id]:
                    object_image_votes[obj_id][frame_idx] = 0
                object_image_votes[obj_id][frame_idx] = count
        
        # select top K frames for each obj
        object_image_votes_topK = {}
        for obj_id in object_image_votes:
            object_image_votes_topK[obj_id] = []
            obj_image_votes_f = object_image_votes[obj_id]
            sorted_frame_idxs = sorted(obj_image_votes_f, key=obj_image_votes_f.get, reverse=True)
            if len(sorted_frame_idxs) > self.top_k:
                object_image_votes_topK[obj_id] = sorted_frame_idxs[:self.top_k]
            else:
                object_image_votes_topK[obj_id] = sorted_frame_idxs
        
        object_ids_in_image_votes = list(object_image_votes_topK.keys())
        for obj_id in object_ids_in_image_votes:
            if obj_id not in list(obj_id_to_label_id_map.keys()):
                del object_image_votes_topK[obj_id]
        
        assert len(list(obj_id_to_label_id_map.keys())) >= len(list(object_image_votes_topK.keys())), 'Mapped < Found'
        
        object_image_embeddings = {}
        for object_id in object_image_votes_topK:
            object_image_votes_topK_frames = object_image_votes_topK[object_id]
            object_image_embeddings[object_id] = {}
            
            for frame_idx in object_image_votes_topK_frames:
                image_path = osp.join(scene_folder, 'sequence', f'frame-{frame_idx}.color.jpg')
                color_img = Image.open(image_path)
                object_image_embeddings[object_id][frame_idx] = self.computeImageFeaturesEachObject(color_img, object_id, object_anno_2D[frame_idx])

        return object_image_embeddings, object_image_votes_topK, object_anno_2D.keys()
    
    def computeImageFeaturesEachObject(self, image: Image.Image, object_id: int, object_anno_2d: np.ndarray) -> np.ndarray:
        object_anno_2d = object_anno_2d.transpose(1, 0)
        object_anno_2d = np.flip(object_anno_2d, 1)
        
        # load image
        image = image.transpose(Image.ROTATE_270)
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
