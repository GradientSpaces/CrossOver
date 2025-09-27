import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, List
import numpy as np
import MinkowskiEngine as ME

from third_party.BLIP.models.blip import blip_feature_extractor
from modules.basic_modules import get_mlp_head
from modules.encoder2D.dinov2 import DinoV2
from modules.layers.pointnet import PointTokenizeEncoder
from modules.build import build_module
from pathlib import Path
from util import point_cloud
from common.constants import ModalityType

class InstanceCrossOverModel(nn.Module):
    def __init__(self, args: DictConfig, device: str = 'cuda') -> None:
        super().__init__()
        self.device = device
        self.out_dim = args.out_dim
        
        # Modality configuration
        self.modalities = ['point', 'cad', 'rgb', 'referral']
        self.feat_dims = {
            ModalityType.POINT: args.input_dim_3d,  # 384
            ModalityType.CAD: args.input_dim_3d,    # 384  
            ModalityType.RGB: args.input_dim_2d,    # 1536
            ModalityType.REF: args.input_dim_1d     # 768
        }
        
        self.point_feature_extractor_name = 'I2PMAE'
        self.point_feature_extractor_ckpt = Path('/drive/pretrained-models/pointbind_i2pmae.pt')
        self.point_feature_extractor = self.loadFeatureExtractor("3D")
        
        self.modality_encoders = nn.ModuleDict({
            ModalityType.POINT: PointTokenizeEncoder(hidden_size=self.feat_dims[ModalityType.POINT]),
            ModalityType.CAD: PointTokenizeEncoder(use_attn=False, hidden_size=self.feat_dims[ModalityType.CAD])
        })
        
        # self.encoder2D = DinoV2('dinov2_vitg14', self.device).feature_extractor
        self.encoder2D = build_module("2D", 'DinoV2', 
                                 ckpt =  'dinov2_vitg14', device = self.device)
        
        self.encoder1D = blip_feature_extractor(pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth', 
                                            image_size=224, vit='large').to(self.device)

        
        self.modality_projections = nn.ModuleDict({
            ModalityType.POINT: get_mlp_head(self.feat_dims[ModalityType.POINT], self.feat_dims[ModalityType.POINT], self.out_dim),
            ModalityType.CAD: get_mlp_head(self.feat_dims[ModalityType.CAD], self.feat_dims[ModalityType.CAD], self.out_dim),
            ModalityType.RGB: get_mlp_head(self.feat_dims[ModalityType.RGB], self.feat_dims[ModalityType.RGB] // 2, self.out_dim),
            ModalityType.REF: get_mlp_head(self.feat_dims[ModalityType.REF], self.feat_dims[ModalityType.REF], self.out_dim)
        })

    def loadFeatureExtractor(self, modality: str) -> torch.nn.Module:
        """Loads and initializes the feature extractor model."""
        model = build_module(modality, self.point_feature_extractor_name)
        state_dict = torch.load(self.point_feature_extractor_ckpt, map_location='cpu')
        state_dict = {k.replace('point_encoder.', '') : v for k, v in state_dict.items() if k.startswith('point_encoder')}
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(self.device)
        
        return model
    
    def encode_point_objects(self, obj_points, obj_masks: torch.Tensor) -> torch.Tensor:
        """Encode raw point cloud data for objects - matches preprocessing approach"""
        
        # obj_points is now a list of raw point clouds (like preprocessing)
        # batch_size is always 1 for single-scan datasets
        if isinstance(obj_points, list):
            # Single batch case - obj_points is list of point clouds
            point_clouds_batch = obj_points
            batch_size = 1
            num_objects = len(point_clouds_batch)
        else:
            # Legacy tensor format (if still used elsewhere)
            batch_size, num_objects, num_points, _ = obj_points.shape
            point_clouds_batch = [obj_points[0, o].cpu().numpy() for o in range(num_objects)]
        
        # Extract features for each object using I2PMAE (like preprocessing)
        object_features = []
        object_locations = []
        
        for o in range(num_objects):
            if obj_masks[0, o]:  
                obj_pts = point_clouds_batch[o] 
                
                sampled_points = point_cloud.sample_and_normalize_pcl(obj_pts)
                object_loc, object_box = point_cloud.get_object_loc_box(obj_pts)
                
                points_pt = torch.from_numpy(sampled_points).unsqueeze(0).to(self.device).float()
                
                with torch.no_grad():
                    obj_feat = self.point_feature_extractor(points_pt) 
                
                object_features.append(obj_feat.squeeze(0))
                object_locations.append(object_loc)  # Use all 6 elements: [center_x, center_y, center_z, size_x, size_y, size_z]
            else:
                object_features.append(torch.zeros(self.feat_dims[ModalityType.POINT]).to(self.device))
                object_locations.append(np.zeros(6))  # Zero location with 6 elements 
        
        point_features = torch.stack(object_features).unsqueeze(0)  # (1, num_objects, feat_dim)
        
        obj_locs = torch.from_numpy(np.stack(object_locations)).unsqueeze(0).to(self.device).float()  # (1, num_objects, 6)
        encoded_features = self.modality_encoders[ModalityType.POINT](point_features, obj_locs, obj_masks)
        
        point_embeddings = self.modality_projections[ModalityType.POINT](encoded_features.view(-1, encoded_features.size(-1)))
        point_embeddings = point_embeddings.view(1, num_objects, -1) 
        return point_embeddings
    
    def encode_cad_objects(self, obj_points, obj_masks: torch.Tensor) -> torch.Tensor:
        """Encode raw CAD point cloud data for objects - matches preprocessing approach"""
        
        # obj_points is now a list of raw CAD point clouds (like preprocessing)
        # batch_size is always 1 for single-scan datasets
        if isinstance(obj_points, list):
            # Single batch case - obj_points is list of CAD point clouds
            cad_clouds_batch = obj_points
            batch_size = 1
            num_objects = len(cad_clouds_batch)
        else:
            # Legacy tensor format (if still used elsewhere)
            batch_size, num_objects, num_points, _ = obj_points.shape
            cad_clouds_batch = [obj_points[0, o].cpu().numpy() for o in range(num_objects)]
        
        # Extract features for each object using I2PMAE (like preprocessing)
        object_features = []
        object_locations = []
        
        for o in range(num_objects):
            if obj_masks[0, o]:  # Only process valid CAD objects
                obj_pts = cad_clouds_batch[o]  # Raw CAD point cloud (variable size)
                
                if len(obj_pts) > 0:  # Check if CAD data exists
                    # Sample and normalize like preprocessing normalizeObjectPCLAndExtractFeats
                    sampled_points = point_cloud.sample_and_normalize_pcl(obj_pts)
                    # Get object location like preprocessing
                    object_loc, object_box = point_cloud.get_object_loc_box(obj_pts)
                    
                    points_pt = torch.from_numpy(sampled_points).unsqueeze(0).to(self.device).float()
                    
                    with torch.no_grad():
                        obj_feat = self.point_feature_extractor(points_pt)  # (1, feat_dim)
                    
                    object_features.append(obj_feat.squeeze(0))
                    object_locations.append(object_loc)  # Use all 6 elements: [center_x, center_y, center_z, size_x, size_y, size_z]
                else:
                    # Empty CAD data
                    object_features.append(torch.zeros(self.feat_dims[ModalityType.CAD]).to(self.device))
                    object_locations.append(np.zeros(6))  # Zero location for empty CAD
            else:
                object_features.append(torch.zeros(self.feat_dims[ModalityType.CAD]).to(self.device))
                object_locations.append(np.zeros(6))  # Zero location for invalid objects
        
        cad_features = torch.stack(object_features).unsqueeze(0)  # (1, num_objects, feat_dim)
        
        # Apply modality encoder with computed object locations (like preprocessing)
        obj_locs = torch.from_numpy(np.stack(object_locations)).unsqueeze(0).to(self.device).float()  # (1, num_objects, 6)
        encoded_features = self.modality_encoders[ModalityType.CAD](cad_features, obj_locs, obj_masks)
        
        cad_embeddings = self.modality_projections[ModalityType.CAD](encoded_features.view(-1, encoded_features.size(-1)))
        cad_embeddings = cad_embeddings.view(1, num_objects, -1)  # (1, num_objects, out_dim)
        return cad_embeddings
    
    def encode_rgb_objects(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Encode RGB images for objects exactly like preprocessing extractFeatures()"""
        num_objects, num_crops = rgb_images.size(1), rgb_images.size(2)
        
        object_embeddings = []
        
        for o in range(num_objects):
            object_crops = rgb_images[0, o]  
            crops_tensor = object_crops.to(self.device)
            
            with torch.no_grad():
                
                features = self.encoder2D.feature_extractor(crops_tensor) 
                cls_tokens = features[:, 0, :] 
                
                mean_features = cls_tokens.mean(dim=0)  
                
            projected_embedding = self.modality_projections[ModalityType.RGB](mean_features.unsqueeze(0)).squeeze(0)
            object_embeddings.append(projected_embedding)
        
        return torch.stack(object_embeddings).unsqueeze(0)
    
    def encode_referral_objects(self, referral_texts: List[List[List[str]]]) -> torch.Tensor:
        """Encode referral texts for objects"""
        batch_embeddings = []
        
        for batch_texts in referral_texts:
            object_embeddings = []
            for obj_referrals in batch_texts:
                if not obj_referrals or (len(obj_referrals) == 1 and not obj_referrals[0].strip()):
                    object_embeddings.append(torch.zeros(self.out_dim).to(self.device))
                    continue
                
                referral_feats = []
                for text in obj_referrals:
                    if not text.strip():  # Skip empty text
                        continue
                    
                    encoded_text = self.encoder1D.tokenizer(text, padding=True, add_special_tokens=True, return_tensors="pt").to(self.device)
                    if encoded_text['input_ids'].shape[1] > 512:
                        continue
                    
                    with torch.no_grad():
                        encoded_text = self.encoder1D.text_encoder(encoded_text.input_ids, attention_mask=encoded_text.attention_mask,
                                                                 return_dict=True, mode='text').last_hidden_state[:, 0]  # (1, 768)
                        referral_feats.append(encoded_text.cpu().numpy().reshape(1, -1)) 
                
                if referral_feats:
                    referral_feats = np.concatenate(referral_feats) 
                    mean_feats = np.mean(referral_feats, axis=0).reshape(1, -1)  
                    
                    mean_embedding = torch.from_numpy(mean_feats.squeeze(0)).to(self.device).float()
                    projected_embedding = self.modality_projections[ModalityType.REF](mean_embedding.unsqueeze(0)).squeeze(0)
                    object_embeddings.append(projected_embedding)
                else:
                    object_embeddings.append(torch.zeros(self.out_dim).to(self.device))
            
            batch_embeddings.append(torch.stack(object_embeddings))
        
        return torch.stack(batch_embeddings)
    

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        embedding_dict = {'embeddings': {}, 'masks': {}}
        
        if 'objects' in data_dict and 'inputs' in data_dict['objects']:
            objects_inputs = data_dict['objects']['inputs']
            masks = data_dict.get('masks', {})
            
            # Point modality
            if 'point' in objects_inputs:
                # objects_inputs['point'] is now a list of point clouds
                if isinstance(objects_inputs['point'], list):
                    num_objects = len(objects_inputs['point'])
                    point_mask = masks.get('point', torch.ones(1, num_objects)).to(self.device)
                else:
                    # Legacy tensor format
                    point_mask = masks.get('point', torch.ones(objects_inputs['point'].size(0), objects_inputs['point'].size(1))).to(self.device)
                
                embedding_dict['embeddings']['point'] = self.encode_point_objects(
                    objects_inputs['point'],  # Pass directly (list or tensor)
                    point_mask
                )
                embedding_dict['masks']['point'] = point_mask
            
            # CAD modality
            if 'cad' in objects_inputs:
                # objects_inputs['cad'] is now a list of CAD point clouds
                if isinstance(objects_inputs['cad'], list):
                    num_objects = len(objects_inputs['cad'])
                    cad_mask = masks.get('cad', torch.ones(1, num_objects)).to(self.device)
                else:
                    # Legacy tensor format
                    cad_mask = masks.get('cad', torch.ones(objects_inputs['cad'].size(0), objects_inputs['cad'].size(1))).to(self.device)
                
                embedding_dict['embeddings']['cad'] = self.encode_cad_objects(
                    objects_inputs['cad'],  # Pass directly (list or tensor)
                    cad_mask
                )
                embedding_dict['masks']['cad'] = cad_mask
            
            # RGB modality
            if 'rgb' in objects_inputs:
                rgb_mask = masks.get('rgb', torch.ones(objects_inputs['rgb'].size(0), objects_inputs['rgb'].size(1))).to(self.device)
                embedding_dict['embeddings']['rgb'] = self.encode_rgb_objects(objects_inputs['rgb'].to(self.device))
                embedding_dict['masks']['rgb'] = rgb_mask
            
            # Referral modality
            if 'referral_texts' in data_dict:
                referral_embed = self.encode_referral_objects(data_dict['referral_texts'])
                if referral_embed is not None:
                    embedding_dict['embeddings']['referral'] = referral_embed
                    num_objects = referral_embed.size(1)
                    embedding_dict['masks']['referral'] = torch.ones(1, num_objects).to(self.device)
        
        
        return embedding_dict