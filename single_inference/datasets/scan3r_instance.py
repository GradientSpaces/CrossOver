import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tvf
from typing import List, Dict
from util.image import mask2box_multi_level
from common import load_utils
from util import scan3r

class Scan3RInstanceInference:
    """Dataset class that loads instance-level data for one specific Scan3R scan"""
    
    def __init__(self, data_dir, process_dir, scan_id, image_size=[224, 224], max_objects=150, max_points_per_object=1024):
        self.scan_id = scan_id
        self.data_dir = data_dir
        self.process_dir = process_dir
        self.image_size = image_size
        self.max_objects = max_objects
        self.max_points_per_object = max_points_per_object
        
        self.top_k = 15  # Top K frames per object
        self.num_levels = 3  # Multi-level cropping levels
        
        # Image transform
        self.image_transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.scans_dir = osp.join(data_dir, 'scans')
        self.files_dir = osp.join(data_dir, 'files')
        self.processed_scans_dir = osp.join(process_dir, 'scans')
        
        referral_path = osp.join(self.files_dir, 'sceneverse/ssg_ref_rel2_template.json')
        self.referrals = []
        self.referrals = load_utils.load_json(referral_path)
        
        
        self.image_transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
        ])
        
        self.model_image_size = tuple(image_size)
        self.top_k = 5  # Top-K frames per object
        self.num_levels = 3  # Multi-level cropping levels
        self.undefined = 0  # Undefined object ID
    
    def load_scan_data(self):
        """Load raw scan data and extract objects for the specific scan"""
        scene_folder = osp.join(self.data_dir,'scans')
        
        ply_data = scan3r.load_ply_data(scene_folder, self.scan_id, 'labels.instances.align.annotated.v2.ply')
        
        vertices = np.stack([ply_data['x'], ply_data['y'], ply_data['z']]).transpose((1, 0))
        object_ids = ply_data['objectId']
        
        unique_object_ids = np.unique(object_ids)
        unique_object_ids = unique_object_ids[unique_object_ids != 0]  # Remove undefined
        
        objects_data = {}
        for obj_id in unique_object_ids:
            mask = object_ids == obj_id
            obj_vertices = vertices[mask]
            
            objects_data[obj_id] = {
                'points': obj_vertices  # Raw points, no sampling/padding
            }
        
        return objects_data, list(objects_data.keys())

    def extract_object_images(self, object_ids: List[int]) -> Dict[int, List[torch.Tensor]]:
        """Extract object images using the same approach as preprocess/feat2D/scan3r.py"""
        scene_folder = os.path.join(self.data_dir, 'scans', self.scan_id)
        color_path = os.path.join(scene_folder, 'sequence')
        
        scene_out_dir = os.path.join(self.process_dir, 'scans', self.scan_id) if self.process_dir else None
        object_anno_2D = np.load(os.path.join(scene_out_dir, 'gt-projection-seg.npz'), allow_pickle=True)

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
        
        object_images = {}
        for object_id in object_ids:
            if object_id not in object_image_votes_topK:
                object_images[object_id] = []
                continue
                
            object_image_crops = []
            topK_frames = object_image_votes_topK[object_id]
            
            for frame_idx in topK_frames:
                color_file = os.path.join(color_path, f'frame-{frame_idx}.color.jpg')
                if not os.path.exists(color_file):
                    continue
                    
                color_img = Image.open(color_file)
                object_anno = object_anno_2D[frame_idx]
                
                frame_crops = self.computeImageFeaturesEachObject(color_img, object_id, object_anno)
                object_image_crops.extend(frame_crops)
            
            object_images[object_id] = object_image_crops
        
        return object_images

    def computeImageFeaturesEachObject(self, image: Image.Image, object_id: int, 
                                     object_anno_2d: np.ndarray) -> List[torch.Tensor]:
        """Multi-level object cropping exactly like preprocess/feat2D/scan3r.py"""
        
        object_anno_2d = object_anno_2d.transpose(1, 0)
        object_anno_2d = np.flip(object_anno_2d, 1)
        image = image.transpose(Image.ROTATE_270)
        
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
        
        scan_referrals = [ref for ref in self.referrals if ref.get('scan_id') == self.scan_id]
        
        for obj_id in object_ids:
            obj_referrals = [ref['utterance'] for ref in scan_referrals if int(ref.get('target_id', -1)) == obj_id - 1]
            object_referrals[obj_id] = obj_referrals if obj_referrals else ['']
        
        return object_referrals
    
    def get_data(self):
        """Return the instance-level data dict for the single scan"""
        objects_data, object_ids = self.load_scan_data()
        
        object_ids = object_ids[:self.max_objects]
        num_objects = len(object_ids)
        
        # Prepare object data
        objects_dict = {'inputs': {}, 'object_locs': {}}
        masks = {}
        
        point_coords = np.zeros((1, num_objects, self.max_points_per_object, 3))
        point_masks = np.zeros((1, num_objects))
        
        for i, obj_id in enumerate(object_ids):
            obj_data = objects_data[obj_id]
            point_coords[0, i] = obj_data['points']  # Raw point coordinates
            point_masks[0, i] = 1.0
        
        objects_dict['inputs']['point'] = torch.from_numpy(point_coords).float()
        masks['point'] = torch.from_numpy(point_masks).bool()
        
        object_images_dict = self.extract_object_images(object_ids)
        
        max_crops = max([len(imgs) for imgs in object_images_dict.values()]) if object_images_dict else 1
        max_crops = max(max_crops, 1)  # Ensure at least 1 crop
        
        rgb_data = torch.zeros(1, num_objects, max_crops, 3, self.image_size[0], self.image_size[1])
        rgb_masks = torch.zeros(1, num_objects).bool()
        
        for i, obj_id in enumerate(object_ids):
            if obj_id in object_images_dict and len(object_images_dict[obj_id]) > 0:
                crops = object_images_dict[obj_id]
                for j, crop in enumerate(crops[:max_crops]):  
                    rgb_data[0, i, j] = crop  
                rgb_masks[0, i] = True
        
        objects_dict['inputs']['rgb'] = rgb_data
        masks['rgb'] = rgb_masks
        
        # Referral data
        object_referrals = self.get_object_referrals(object_ids)
        batch_referral_texts = []
        referral_masks = np.zeros((1, len(object_ids)), dtype=bool) 
        
        for i, obj_id in enumerate(object_ids):
            obj_referrals = object_referrals.get(obj_id, [''])
            valid_referrals = [ref for ref in obj_referrals if ref.strip()]
            if valid_referrals:
                batch_referral_texts.append(valid_referrals)  
                referral_masks[0, i] = True
            else:
                batch_referral_texts.append([''])  
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
    