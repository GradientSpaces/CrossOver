import os.path as osp
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
import shutil
from PIL import Image
from scipy.spatial.transform import Rotation as R
import cv2
from common import load_utils
from util import render, structured3d, visualisation
from util import image as image_util
import os
from preprocess.build import PROCESSOR_REGISTRY
from preprocess.feat2D.base import Base2DProcessor


@PROCESSOR_REGISTRY.register()
class Structured3D_2DProcessor(Base2DProcessor):
    def __init__(self, config_data, config_2D, split) -> None:
        super(Structured3D_2DProcessor, self).__init__(config_data, config_2D, split)
        self.data_dir = config_data.base_dir
        files_dir = osp.join(config_data.base_dir, 'files')
        self.split = split
        
        self.scan_ids = []
        self.scan_ids = structured3d.get_scan_ids(files_dir, split)
        
        self.out_dir = config_data.process_dir
        load_utils.ensure_dir(self.out_dir)
        
        self.model_image_size = config_2D.image.model_size
        
        self.frame_skip = config_data.skip_frames
        self.top_k = config_2D.image.top_k
        self.num_levels = config_2D.image.num_levels
        
        
        # get frame_indexes
        self.frame_pose_data = {}
        for scan_id in self.scan_ids:
            full_scan_id = scan_id
            scan_id = scan_id.split('_')
            room_id = scan_id[-1]
            scan_id = scan_id[0]+'_'+scan_id[1]
            scene_folder = osp.join(self.data_dir, 'scans', scan_id, '2D_rendering', room_id, 'perspective', 'full')
            frame_idxs = [f for f in os.listdir(scene_folder) if f[0] != '.' and f[0] != 'g']
            pose_data = structured3d.load_all_poses(scene_folder, frame_idxs)
            self.frame_pose_data[full_scan_id] = pose_data


    def compute2DFeatures(self):
        for scan_id in tqdm(self.scan_ids):
            self.compute2DImagesAndSeg(scan_id)
            self.compute2DFeaturesEachScan(scan_id)
            # if self.split == 'val':
            #     self.computeAllImageFeaturesEachScan(scan_id)
    
    def compute2DImagesAndSeg(self, scan_id):
        full_scan_id = scan_id
        scan_id = scan_id.split('_')
        room_id = scan_id[-1]
        scan_id = scan_id[0]+'_'+scan_id[1]
        scene_folder = osp.join(self.data_dir, 'scans', scan_id,'2D_rendering', room_id, 'perspective', 'full')
        
        obj_id_imgs = {}
        for frame_idx in self.frame_pose_data[full_scan_id]:
            image_path=osp.join(scene_folder, frame_idx, 'instance.png')
            obj_id_map = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            obj_id_imgs[frame_idx] = obj_id_map

        if osp.exists(osp.join(scene_folder, 'gt-projection')):
            shutil.rmtree(osp.join(scene_folder, 'gt-projection'))
    
        torch.save(obj_id_imgs, osp.join(scene_folder, 'gt-projection-seg.pt'))
    
    def compute2DFeaturesEachScan(self, scan_id):
        full_scan_id = scan_id
        scan_id = scan_id.split('_')
        room_id = scan_id[-1]
        scan_id = scan_id[0]+'_'+scan_id[1]
        scene_folder = osp.join(self.data_dir, 'scans', scan_id,'2D_rendering', room_id, 'perspective', 'full')
        
        scene_out_dir = osp.join(self.out_dir, full_scan_id)
        load_utils.ensure_dir(scene_out_dir)
        
        obj_id_to_label_id_map = torch.load(osp.join(scene_out_dir, 'object_id_to_label_id_map.pt'))['obj_id_to_label_id_map']
        
        floorplan_img_path = osp.join(self.data_dir,'scans', scan_id, 'floorplans', f'{room_id}.png')
        floorplan_img = cv2.imread(floorplan_img_path)
        floorplan_img = cv2.cvtColor(floorplan_img, cv2.COLOR_BGR2RGB)
        floorplan_img = cv2.cvtColor(floorplan_img, cv2.COLOR_RGB2GRAY)
        floorplan_img = cv2.cvtColor(floorplan_img, cv2.COLOR_GRAY2RGB)
        floorplan_img = image_util.crop_image(floorplan_img, floorplan_img_path.replace('.png', '_cropped.png'))
        floorplan_embeddings = None
        
        if floorplan_img is not None:
            floorplan_img = floorplan_img.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
            floorplan_img_pt = self.model.base_tf(floorplan_img)
            floorplan_embeddings = self.extractFeatures([floorplan_img_pt], return_only_cls_mean = False)            
        floorplan_dict = {'img' : floorplan_img, 'embedding' : floorplan_embeddings}
        # print(floorplan_dict)
        # Multi-view Image -- Object (Embeddings)
        object_image_embeddings, object_image_votes_topK, frame_idxs = self.computeImageFeaturesAllObjectsEachScan(scene_folder, obj_id_to_label_id_map)
        
        # Multi-view Image -- Scene (Images + Embeddings)
        frame_idxs = list(self.frame_pose_data[full_scan_id].keys())
        pose_data, scene_images_pt, scene_image_embeddings, sampled_frame_idxs = self.computeSelectedImageFeaturesEachScan(full_scan_id, scene_folder, frame_idxs)
        
        # Visualise
        # camera_info = structured3d.load_intrinsics(scene_folder)
        # intrinsic_mat = camera_info['intrinsic_mat']
        
        # scene_mesh = o3d.io.read_triangle_mesh(osp.join(self.data_dir, 'scans', scan_id, '3D_rendering', room_id,'room_mesh.ply'))
        # intrinsics = { 'f' : intrinsic_mat[0, 0], 'cx' : intrinsic_mat[0, 2], 'cy' : intrinsic_mat[1, 2], 
        #                 'w' : int(camera_info['width']), 'h' : int(camera_info['height'])}
        
        # cams_visualised_on_mesh = visualisation.visualise_camera_on_mesh(scene_mesh, pose_data[sampled_frame_idxs], intrinsics, stride=1)
        # image_path = osp.join(scene_out_dir, 'sel_cams_on_mesh.png')
        # Image.fromarray((cams_visualised_on_mesh * 255).astype(np.uint8)).save(image_path)
        
        data2D = {}
        data2D['objects'] = {'image_embeddings': object_image_embeddings, 'topK_images_votes' : object_image_votes_topK}
        data2D['scene']   = {'scene_embeddings': scene_image_embeddings, 'images' : scene_images_pt, 
                                'frame_idxs' : frame_idxs, 'sampled_cam_idxs' : sampled_frame_idxs}
        
        data2D['scene']['floorplan'] = floorplan_dict
        
        torch.save(data2D, osp.join(scene_out_dir, 'data2D.pt'))
    
    # def computeAllImageFeaturesEachScan(self, scan_id):
    #     scene_folder = osp.join(self.data_dir, 'scenes', scan_id)
    #     color_path = osp.join(scene_folder, 'sequence')
    #     scene_out_dir = osp.join(self.out_dir, scan_id)
    #     load_utils.ensure_dir(scene_out_dir)
        
    #     frame_idxs = list(self.frame_pose_data[scan_id].keys())
        
    #     # Extract Scene Image Features
    #     scene_images_pt = []
    #     scene_image_embeddings = []
    #     for frame_index in frame_idxs:
    #         image = Image.open(osp.join(color_path, f'frame-{frame_index}.color.jpg'))
    #         image = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
    #         image_pt = self.model.base_tf(image)
    #         # image_pt = torch.zeros(1, 1536)
            
    #         scene_image_embeddings.append(self.extractFeatures([image_pt], return_only_cls_mean= False))
    #         scene_images_pt.append(image_pt)
    #     scene_image_embeddings = np.concatenate(scene_image_embeddings)
    #     data2D = {} 
    #     data2D['scene'] = {'scene_embeddings': scene_image_embeddings, 'images' : scene_images_pt, 
    #                        'frame_idxs' : frame_idxs}
    #     torch.save(data2D, osp.join(scene_out_dir, 'data2D_all_images.pt'))
    
    def computeSelectedImageFeaturesEachScan(self, scan_id, color_path, frame_idxs):
        # Sample Camera Indexes Based on Rotation Matrix From Grid
        pose_data = []
        for frame_idx in frame_idxs:
            pose = self.frame_pose_data[scan_id][frame_idx]
            rot_quat = R.from_matrix(pose[:3, :3]).as_quat()
            trans = pose[:3, 3]
            pose_data.append([trans[0], trans[1], trans[2], rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]])
            
        pose_data = np.array(pose_data)
        
        sampled_frame_idxs = image_util.sample_camera_pos_on_grid(pose_data)
        # print(sampled_frame_idxs)
        scene_images_pt = []
        for idx in sampled_frame_idxs:
            frame_index = frame_idxs[idx]
            
            image = Image.open(osp.join(color_path, frame_index, f'rgb_rawlight.png'))
            image = image.convert('RGB')
            image = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
            image_pt = self.model.base_tf(image)
            scene_images_pt.append(image_pt)
            
        
        scene_image_embeddings = self.extractFeatures(scene_images_pt, return_only_cls_mean= False)
        
        return pose_data, scene_images_pt, scene_image_embeddings, sampled_frame_idxs
    
    def computeImageFeaturesAllObjectsEachScan(self, scene_folder, obj_id_to_label_id_map):
        object_anno_2D = torch.load(osp.join(scene_folder, 'gt-projection-seg.pt'))
        object_image_votes = {}
        
        # iterate over all frames
        for frame_idx in object_anno_2D:
            obj_2D_anno_frame = object_anno_2D[frame_idx]
            # process 2D anno
            obj_ids, counts = np.unique(obj_2D_anno_frame, return_counts=True)
            for idx in range(len(obj_ids)):
                obj_id = obj_ids[idx]
                count = counts[idx]
                
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
                image_path = osp.join(scene_folder, frame_idx, 'rgb_rawlight.png')
                # print(image_path)
                color_img = Image.open(image_path)
                # print(color_img.mode)
                color_img = color_img.convert('RGB')
                object_image_embeddings[object_id][frame_idx] = self.computeImageFeaturesEachObject(color_img, object_id, object_anno_2D[frame_idx])

        return object_image_embeddings, object_image_votes_topK, object_anno_2D.keys()
    
    def computeImageFeaturesEachObject(self, image, object_id, object_anno_2d):
        # print(np.array(image).shape)
        object_anno_2d = object_anno_2d.transpose(1, 0)
        object_anno_2d = np.flip(object_anno_2d, 1)
        
        # load image
        object_mask = object_anno_2d == object_id
        
        images_crops = []
        for level in range(self.num_levels):
            mask_tensor = torch.from_numpy(object_mask).float()
            x1, y1, x2, y2 = image_util.mask2box_multi_level(mask_tensor, level)
            cropped_img = image.crop((x1, y1, x2, y2))
            # print(np.array(cropped_img).shape)
            cropped_img = cropped_img.resize((self.model_image_size[1], self.model_image_size[1]), Image.BICUBIC)
            img_pt = self.model.base_tf(cropped_img)
            images_crops.append(img_pt)
            # images_crops.append(cropped_img)
            
        
        if(len(images_crops) > 0):
            mean_feats = self.extractFeatures(images_crops, return_only_cls_mean = True)
        return mean_feats
        
