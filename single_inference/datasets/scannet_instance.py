import os
import os.path as osp
import numpy as np
import torch
import imageio
import skimage.transform as sktf
from PIL import Image
from torchvision import transforms as tvf
from typing import List, Dict
from util.image import mask2box_multi_level
from util import scannet
from common import load_utils

class ScannetInstanceInference:
    """Dataset class that loads instance-level data for one specific Scannet scan"""
    
    def __init__(self, data_dir, scan_id, image_size=[224, 224], max_objects=150, 
                 voxel_size=0.02, frame_skip=5, max_points_per_object=1024, shape_dir=None):
        self.scan_id = scan_id
        self.data_dir = data_dir
        self.data_root = data_dir
        self.image_size = image_size
        self.max_objects = max_objects
        self.voxel_size = voxel_size
        self.frame_skip = frame_skip
        self.max_points_per_object = max_points_per_object
        
        self.orig_image_size = (968, 1296)  # ScanNet original image size
        self.model_image_size = tuple(image_size)
        self.top_k = 15  # Top-K frames per object
        self.num_levels = 3  # Multi-level cropping levels
        self.undefined = 0  # Undefined object ID
        
        self.scans_dir = osp.join(data_dir, 'scans')
        self.files_dir = osp.join(data_dir, 'files')
        
        referral_path = osp.join(self.files_dir, 'sceneverse/ssg_ref_rel2_template.json')
        self.referrals = []
        if osp.exists(referral_path):
            self.referrals = load_utils.load_json(referral_path)
        
        self.label_map = scannet.read_label_map(self.files_dir, label_from='raw_category', label_to='nyu40id')
        
        cad_path = osp.join(self.files_dir, 'scan2cad_full_annotations.json')
        self.cad_annotations = []
        self.cad_annotations = load_utils.load_json(cad_path)
        
        self.shape_dir = shape_dir
        
        self.base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
        ])
        
        self.image_transform = self.base_tf
        
    
    def load_scan_data(self):
        """Load raw scan data and extract objects for the specific scan"""
        scan_folder = osp.join(self.scans_dir, self.scan_id)
        
        # Load mesh and segmentation data
        mesh_file = osp.join(scan_folder, self.scan_id + '_vh_clean_2.ply')
        aggre_file = osp.join(scan_folder, self.scan_id + '_vh_clean.aggregation.json')
        seg_file = osp.join(scan_folder, self.scan_id + '_vh_clean_2.0.010000.segs.json')
        meta_file = osp.join(scan_folder, self.scan_id + '.txt')
        
        if not all(osp.exists(f) for f in [mesh_file, aggre_file, seg_file]):
            return None, None, None, None
        
        mesh_vertices, _, instance_ids, instance_bboxes, object_id_to_label_id, _ = \
            scannet.export(mesh_file, aggre_file, seg_file, meta_file, self.label_map, 
                            axis_alignment=True, output_file=None)
        
        mesh_points = mesh_vertices[:, 0:3] 
        
        cad_data = {}
        shape_annot = [cad_annot for cad_annot in self.cad_annotations if cad_annot['id_scan'] == self.scan_id]
        if len(shape_annot) > 0:
            shape_annot = shape_annot[0]
            shape_annot_to_instance_map = scannet.get_cad_model_to_instance_mapping(
                instance_bboxes, shape_annot, meta_file, self.shape_dir
            )
            if shape_annot_to_instance_map is not None:
                for instance_id, shape_data in shape_annot_to_instance_map.items():
                    if instance_id in object_id_to_label_id:  # Only include valid instances
                        cad_data[instance_id] = shape_data['points']
        
        return mesh_points, instance_ids, object_id_to_label_id, cad_data
    
    def extract_object_point_clouds(self, mesh_points, instance_ids, object_id_to_label_id):
        """Extract raw point clouds for each object (no sampling - done in model like preprocessing)"""
        object_point_clouds = {}
        
        for instance_id in object_id_to_label_id.keys():
            
            object_mask = instance_ids == instance_id
            object_points = mesh_points[object_mask]
            
            # Store raw points - sampling will be done in model during feature extraction
            # This matches the preprocessing approach where sampling happens in normalizeObjectPCLAndExtractFeats
            object_point_clouds[instance_id] = {
                'points': object_points,  # Raw points, no sampling/padding
                'label_id': object_id_to_label_id[instance_id]
            }
        
        return object_point_clouds
    
    def extract_object_images(self, object_ids: List[int]) -> Dict[int, List[torch.Tensor]]:
        """Extract object images using the same approach as preprocess/feat2D/scannet.py"""
        color_path = os.path.join(self.data_root, 'scans', self.scan_id, 'data', 'color')
        instance_path = os.path.join(self.data_root, 'scans', self.scan_id, 'data', 'instance-filt')
        
        from util.scannet import load_poses
        pose_data = load_poses(os.path.join(self.data_root, 'scans'), self.scan_id, skip=self.frame_skip)
        frame_idxs = list(pose_data.keys())
        
        object_image_votes = {}
        object_anno_2d = {}
        
        for frame_idx in frame_idxs:
            instance_file = os.path.join(instance_path, f'{frame_idx}.png')
            if not os.path.exists(instance_file):
                continue
                
            image = np.array(imageio.imread(instance_file))
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
            obj_image_votes_f = object_image_votes[object_id]
            sorted_frame_idxs = sorted(obj_image_votes_f, key=obj_image_votes_f.get, reverse=True)
            if len(sorted_frame_idxs) > self.top_k:
                object_image_votes_topK[object_id] = sorted_frame_idxs[:self.top_k]
            else:
                object_image_votes_topK[object_id] = sorted_frame_idxs
        
        object_images = {}
        for object_id in object_ids:
            if object_id not in object_image_votes_topK:
                object_images[object_id] = []
                continue
                
            object_image_crops = []
            topK_frames = object_image_votes_topK[object_id]
            
            for frame_idx in topK_frames:
                color_file = os.path.join(color_path, f'{frame_idx}.jpg')
                    
                color_img = Image.open(color_file)
                object_anno = object_anno_2d[frame_idx]
                
                frame_crops = self.computeImageFeaturesEachObject(color_img, object_id, object_anno)
                object_image_crops.extend(frame_crops)
            
            object_images[object_id] = object_image_crops
        
        return object_images
    
    def computeImageFeaturesEachObject(self, image: Image.Image, object_id: int, 
                                     object_anno_2d: np.ndarray) -> List[torch.Tensor]:
        """Multi-level object cropping exactly like preprocess/feat2D/scannet.py"""
        
        object_mask = object_anno_2d == object_id
        
        images_crops = []
        for level in range(self.num_levels):
            mask_tensor = torch.from_numpy(object_mask).float()
            x1, y1, x2, y2 = mask2box_multi_level(mask_tensor, level)
            cropped_img = image.crop((x1, y1, x2, y2))
            cropped_img = cropped_img.resize((self.model_image_size[1], self.model_image_size[1]), Image.BICUBIC)
            
            img_tensor = self.image_transform(cropped_img)
            images_crops.append(img_tensor)
        
        return images_crops
    
    def get_object_referrals(self, object_ids):
        """Get referral texts for objects"""
        object_referrals = {}
        
        scan_referrals = [ref for ref in self.referrals if ref['scan_id'] == self.scan_id]
        
        for obj_id in object_ids:
            obj_referrals = [ref['utterance'] for ref in scan_referrals if int(ref['target_id']) == obj_id - 1]
            object_referrals[obj_id] = obj_referrals if obj_referrals else ['']
        
        return object_referrals
    
    def get_data(self):
        """Return the instance-level data dict for the single scan"""
        mesh_points, instance_ids, object_id_to_label_id, cad_data = self.load_scan_data()
        object_point_clouds = self.extract_object_point_clouds(mesh_points, instance_ids, object_id_to_label_id)
        
        object_ids = list(object_point_clouds.keys())[:self.max_objects]
        num_objects = len(object_ids)
        
        objects_dict = {'inputs': {}, 'object_locs': {}}
        masks = {}
        
        # Store raw point clouds - model will handle sampling during feature extraction
        point_clouds = []
        point_masks = []
        
        for i, obj_id in enumerate(object_ids):
            obj_data = object_point_clouds[obj_id]
            point_clouds.append(obj_data['points'])  # Raw points (variable size)
            point_masks.append(True)
        
        objects_dict['inputs']['point'] = point_clouds  # List of raw point clouds
        masks['point'] = torch.tensor([[True] * num_objects]).bool()  # (1, num_objects)
        
        object_images_dict = self.extract_object_images(object_ids)
        
        max_views = max([len(imgs) for imgs in object_images_dict.values()]) if object_images_dict else 1
        max_views = max(max_views, 1)  # Ensure at least 1 view
        
        rgb_data = torch.zeros(1, num_objects, max_views, 3, self.image_size[0], self.image_size[1])
        rgb_masks = torch.zeros(1, num_objects).bool()
        
        for i, obj_id in enumerate(object_ids):
            if obj_id in object_images_dict and object_images_dict[obj_id]:
                obj_images = object_images_dict[obj_id]
                num_obj_views = min(len(obj_images), max_views)
                
                for v in range(num_obj_views):
                    rgb_data[0, i, v] = obj_images[v]
                
                rgb_masks[0, i] = True
            else:
                # Create dummy data for objects without RGB images
                rgb_masks[0, i] = False
        
        objects_dict['inputs']['rgb'] = rgb_data
        masks['rgb'] = rgb_masks
        
        if cad_data:
            # Store raw CAD point clouds - model will handle sampling during feature extraction
            cad_point_clouds = []
            cad_mask_list = []
            
            for i, obj_id in enumerate(object_ids):
                if obj_id in cad_data:
                    cad_points = cad_data[obj_id]  # Raw CAD points (variable size)
                    cad_point_clouds.append(cad_points)
                    cad_mask_list.append(True)
                else:
                    # No CAD data for this object - add dummy data
                    cad_point_clouds.append(np.empty((0, 3)))  # Empty point cloud
                    cad_mask_list.append(False)
            
            objects_dict['inputs']['cad'] = cad_point_clouds  # List of raw CAD point clouds
            masks['cad'] = torch.tensor([cad_mask_list]).bool()  # (1, num_objects)
        
        object_referrals = self.get_object_referrals(object_ids)
        batch_referral_texts = []
        referral_masks = np.zeros((1, len(object_ids)), dtype=bool)  # [batch, objects]
        
        for i, obj_id in enumerate(object_ids):
            obj_referrals = object_referrals.get(obj_id, [''])
            valid_referrals = [ref for ref in obj_referrals if ref.strip()]
            if valid_referrals:
                batch_referral_texts.append(valid_referrals)  
                referral_masks[0, i] = True
            else:
                batch_referral_texts.append(['']) 
                referral_masks[0, i] = False
        
        referral_texts = [batch_referral_texts]  
        masks['referral'] = torch.from_numpy(referral_masks).bool()
        
        return {
            'scan_id': self.scan_id,
            'objects': objects_dict,
            'masks': masks,
            'referral_texts': referral_texts,
            'object_ids': object_ids
        }