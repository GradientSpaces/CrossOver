import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as tvf
import random
from typing import List, Dict

from common import load_utils
from util import arkit

class ARKitScenesSingleScanDataset:
    """Dataset class that loads instance-level data for one specific ARKitScenes scan"""
    
    def __init__(self, data_dir, process_dir, scan_id, image_size=[224, 224], max_objects=150, max_points_per_object=1024):
        self.scan_id = scan_id
        self.data_dir = data_dir
        self.process_dir = process_dir
        self.image_size = image_size
        self.model_image_size = image_size  # For compatibility with computeImageFeaturesEachObject
        self.max_objects = max_objects
        self.max_points_per_object = max_points_per_object
        
        # Image processing parameters from feat2D
        self.top_k = 15  # Top K frames per object
        self.num_levels = 3  # Multi-level cropping levels
        
        # Image transform
        self.image_transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # ARKitScenes uses standard 'scans' directory - exactly like the original instance dataset
        self.scans_dir = osp.join(data_dir, 'scans')
        self.files_dir = osp.join(data_dir, 'files')
        self.processed_scans_dir = osp.join(process_dir, 'scans')
        
        # Load referrals if available - exactly like the original  
        referral_path = osp.join(self.files_dir, 'sceneverse/ssg_ref_rel2_template.json')
        self.referrals = []
        if osp.exists(referral_path):
            self.referrals = load_utils.load_json(referral_path)
        
        # Load objects if available - exactly like the original
        objects_path = osp.join(self.files_dir, 'objects.json')
        self.objects = []
        if osp.exists(objects_path):
            self.objects = load_utils.load_json(objects_path).get('scans', [])
        
        self.image_transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
        ])
        
        # Parameters matching preprocess/feat2D/arkit.py - exactly like the original
        self.orig_image_size = (192, 256)  # ARKit image size
        self.model_image_size = tuple(image_size)
        self.top_k = 5  # Top-K frames per object
        self.num_levels = 3  # Multi-level cropping levels
        self.undefined = 0  # Undefined object ID
    
    def load_scan_data(self):
        """Load raw scan data and extract objects for the specific scan"""
        # ARKitScenes uses 'scans' directory structure - exactly like the original instance dataset
        scene_folder = osp.join(self.data_dir, 'scans', self.scan_id)
        
        # Load 3D object annotations - exactly like the original
        objects_path = osp.join(scene_folder, f"{self.scan_id}_3dod_annotation.json")
        if not osp.exists(objects_path):
            return None, None
            
        annotations = load_utils.load_json(objects_path)
        
        # Load PLY data with object IDs using ARKit utilities - exactly like the original
        from util import arkit
        ply_data = arkit.load_ply_data(osp.join(self.data_dir, 'scans'), self.scan_id, annotations)
        
  
        
        # Extract object information - exactly like the original
        # vertices = ply_data['vertices']
        vertices = np.stack([ply_data['x'], ply_data['y'], ply_data['z']]).transpose((1, 0))
        object_ids = ply_data['objectId']
        
        # Get unique object IDs
        unique_object_ids = np.unique(object_ids)
        unique_object_ids = unique_object_ids[unique_object_ids != 0]  # Remove undefined
        
        # Extract per-object point clouds
        objects_data = {}
        for obj_id in unique_object_ids:
            mask = object_ids == obj_id
            obj_vertices = vertices[mask]
            
            if len(obj_vertices) > 10:  # Skip objects with too few points
                # Subsample or pad points
                if len(obj_vertices) > self.max_points_per_object:
                    indices = np.random.choice(len(obj_vertices), self.max_points_per_object, replace=False)
                    obj_vertices = obj_vertices[indices]
                elif len(obj_vertices) < self.max_points_per_object:
                    pad_size = self.max_points_per_object - len(obj_vertices)
                    obj_vertices = np.pad(obj_vertices, ((0, pad_size), (0, 0)), mode='constant')
                
                objects_data[obj_id] = {
                    'points': obj_vertices
                }
        
        return objects_data, list(objects_data.keys())

    def extract_object_images(self, object_ids: List[int]) -> Dict[int, List[torch.Tensor]]:
        """Extract object images using the same approach as preprocess/feat2D/arkit.py"""
        scene_folder = os.path.join(self.data_dir, 'scans', self.scan_id)
        color_path = os.path.join(scene_folder, f'{self.scan_id}_frames', 'lowres_wide')
        
        # Load projected segmentation from preprocessing (like ARKit does)
        scene_out_dir = os.path.join(self.process_dir, 'scans', self.scan_id) if self.process_dir else None
        if scene_out_dir and os.path.exists(os.path.join(scene_out_dir, 'gt-projection-seg.npz')):
            object_anno_2D = np.load(os.path.join(scene_out_dir, 'gt-projection-seg.npz'), allow_pickle=True)
        else:
            # If no preprocessing, return empty results
            return {obj_id: [] for obj_id in object_ids}
        
        # First pass: count object pixels per frame (like preprocessing)
        object_image_votes = {}
        for frame_idx in object_anno_2D:
            obj_2D_anno_frame = object_anno_2D[frame_idx]
            obj_ids, counts = np.unique(obj_2D_anno_frame, return_counts=True)
            
            for idx in range(len(obj_ids)):
                obj_id = obj_ids[idx]
                count = counts[idx]
                if obj_id == 0:  # undefined
                    continue
                    
                if obj_id not in object_image_votes:
                    object_image_votes[obj_id] = {}
                object_image_votes[obj_id][frame_idx] = count
        
        # Second pass: get top-K frames per object
        object_image_votes_topK = {}
        for object_id in object_ids:
            if object_id not in object_image_votes:
                continue
                
            obj_image_votes_f = object_image_votes[object_id]
            sorted_frame_idxs = sorted(obj_image_votes_f, key=obj_image_votes_f.get, reverse=True)
            if len(sorted_frame_idxs) > self.top_k:
                object_image_votes_topK[object_id] = sorted_frame_idxs[:self.top_k]
            else:
                object_image_votes_topK[object_id] = sorted_frame_idxs
        
        # Third pass: extract features using multi-level cropping
        object_images = {}
        for object_id in object_ids:
            if object_id not in object_image_votes_topK:
                object_images[object_id] = []
                continue
                
            object_image_crops = []
            topK_frames = object_image_votes_topK[object_id]
            
            for frame_idx in topK_frames:
                color_file = os.path.join(color_path, f'{self.scan_id}_{frame_idx}.png')
                if not os.path.exists(color_file):
                    continue
                    
                color_img = Image.open(color_file)
                object_anno = object_anno_2D[frame_idx]
                
                # Multi-level cropping like ARKit preprocessing
                frame_crops = self.computeImageFeaturesEachObject(color_img, object_id, object_anno)
                object_image_crops.extend(frame_crops)
            
            object_images[object_id] = object_image_crops
        
        return object_images

    def computeImageFeaturesEachObject(self, image: Image.Image, object_id: int, 
                                     object_anno_2d: np.ndarray) -> List[torch.Tensor]:
        """Multi-level object cropping exactly like preprocess/feat2D/arkit.py"""
        from util.image import mask2box_multi_level
        
        # Apply ARKit specific transformations
        object_anno_2d = object_anno_2d.transpose(1, 0)
        object_anno_2d = np.flip(object_anno_2d, 1)
        
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
        
        # Get referrals for this scan
        scan_referrals = [ref for ref in self.referrals if ref.get('scan_id') == self.scan_id]
        
        for obj_id in object_ids:
            obj_referrals = [ref['utterance'] for ref in scan_referrals if int(ref.get('target_id', -1)) == obj_id - 1]
            object_referrals[obj_id] = obj_referrals if obj_referrals else ['']
        
        return object_referrals
    
    def get_data(self):
        """Return the instance-level data dict for the single scan"""
        # Load raw scan data
        objects_data, object_ids = self.load_scan_data()
        
        if objects_data is None or len(object_ids) == 0:
            return None
        
        # Limit number of objects
        object_ids = object_ids[:self.max_objects]
        num_objects = len(object_ids)
        
        # Prepare object data
        objects_dict = {'inputs': {}, 'object_locs': {}}
        masks = {}
        
        # Point cloud data - only coordinates needed (model extracts features internally)
        point_coords = np.zeros((1, num_objects, self.max_points_per_object, 3))
        point_masks = np.zeros((1, num_objects))
        
        for i, obj_id in enumerate(object_ids):
            obj_data = objects_data[obj_id]
            point_coords[0, i] = obj_data['points']  # Raw point coordinates
            point_masks[0, i] = 1.0
        
        objects_dict['inputs']['point'] = torch.from_numpy(point_coords).float()
        masks['point'] = torch.from_numpy(point_masks).bool()
        
        # RGB data - extract object-specific image crops
        object_images_dict = self.extract_object_images(object_ids)
        
        # Process RGB data for each object
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
        
        # Referral data
        object_referrals = self.get_object_referrals(object_ids)
        # Create proper batch structure: [batch][objects][referrals_per_object]
        batch_referral_texts = []
        referral_masks = np.zeros((1, len(object_ids)), dtype=bool)  # [batch, objects]
        
        for i, obj_id in enumerate(object_ids):
            obj_referrals = object_referrals.get(obj_id, [''])
            # Keep all valid referrals for this object (filter out empty strings)
            valid_referrals = [ref for ref in obj_referrals if ref.strip()]
            if valid_referrals:
                batch_referral_texts.append(valid_referrals)  # Keep all referrals
                referral_masks[0, i] = True
            else:
                batch_referral_texts.append([''])  # Empty list with one empty string
                referral_masks[0, i] = False
        
        referral_texts = [batch_referral_texts]  # Wrap in batch dimension
        masks['referral'] = torch.from_numpy(referral_masks).bool()
        
        return {
            'scan_id': self.scan_id,
            'objects': objects_dict,
            'masks': masks,
            'referral_texts': referral_texts,
            'object_ids': object_ids
        }
    