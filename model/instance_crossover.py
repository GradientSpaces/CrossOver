import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, List
import numpy as np
import MinkowskiEngine as ME

from third_party.BLIP.models.blip import blip_feature_extractor
from modules.layers.patch_encoder import Mlps
from modules.basic_modules import get_mlp_head
from modules.encoder2D.dinov2 import DinoV2
from modules.layers.pointnet import PointTokenizeEncoder
from modules.layers.sparse_conv_encoder import ResNet34
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
        
        # 3D feature extractor for raw point clouds (like preprocessing)
        self.point_feature_extractor_name = 'I2PMAE'
        self.point_feature_extractor_ckpt = Path('/drive/pretrained-models/pointbind_i2pmae.pt')
        self.point_feature_extractor = self.loadFeatureExtractor("3D")
        
        # Object-level encoders (matching SceneLevelEncoder naming)
        self.modality_encoders = nn.ModuleDict({
            ModalityType.POINT: PointTokenizeEncoder(hidden_size=self.feat_dims[ModalityType.POINT]),
            ModalityType.CAD: PointTokenizeEncoder(use_attn=False, hidden_size=self.feat_dims[ModalityType.CAD])
        })
        
        # 2D encoder for RGB features
        self.encoder2D = DinoV2('dinov2_vitg14', self.device).feature_extractor
        
        # 1D encoder for referral features  
        self.encoder1D = blip_feature_extractor(pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth', 
                                            image_size=224, vit='large').to(self.device)
        # self.encoder1D = blip_model.text_encoder
        # self.tokenizer = blip_model.tokenizer
        
        # Modality projection heads (matching SceneLevelEncoder naming)
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
    
    def encode_point_objects(self, obj_points: torch.Tensor, obj_masks: torch.Tensor) -> torch.Tensor:
        """Encode raw point cloud data for objects"""
        # obj_points: (batch_size, num_objects, num_points, 3) - raw point coordinates
        # obj_locs: (batch_size, num_objects, num_points, 3)
        # obj_masks: (batch_size, num_objects)
        
        batch_size, num_objects, num_points, _ = obj_points.shape
        
        # Extract features for each object using I2PMAE
        object_features = []
        for b in range(batch_size):
            batch_features = []
            for o in range(num_objects):
                if obj_masks[b, o]:  # Only process valid objects
                    # Get object points and normalize like preprocessing
                    obj_pts = obj_points[b, o].cpu().numpy()  # (num_points, 3)
                    
                    # Sample and normalize point cloud (like preprocessing)
                    sampled_points = point_cloud.sample_and_normalize_pcl(obj_pts)
                    points_pt = torch.from_numpy(sampled_points).unsqueeze(0).to(self.device).float()
                    
                    # Extract features using I2PMAE
                    with torch.no_grad():
                        obj_feat = self.point_feature_extractor(points_pt)  # (1, feat_dim)
                    
                    batch_features.append(obj_feat.squeeze(0))
                else:
                    # Zero features for invalid objects
                    batch_features.append(torch.zeros(self.feat_dims[ModalityType.POINT]).to(self.device))
            
            object_features.append(torch.stack(batch_features))
        
        # Stack all batch features: (batch_size, num_objects, feat_dim)
        point_features = torch.stack(object_features)
        
        # Project to output dimension
        point_embeddings = self.modality_projections[ModalityType.POINT](point_features.view(-1, point_features.size(-1)))
        point_embeddings = point_embeddings.view(batch_size, num_objects, -1)
        return point_embeddings
    
    def encode_cad_objects(self, obj_points: torch.Tensor, obj_masks: torch.Tensor) -> torch.Tensor:
        """Encode raw CAD point cloud data for objects"""
        # obj_points: (batch_size, num_objects, num_points, 3) - raw CAD point coordinates
        # obj_masks: (batch_size, num_objects)
        
        batch_size, num_objects, num_points, _ = obj_points.shape
        
        # Extract features for each CAD object using I2PMAE
        object_features = []
        for b in range(batch_size):
            batch_features = []
            for o in range(num_objects):
                if obj_masks[b, o]:  # Only process valid objects
                    # Get CAD object points and normalize like preprocessing
                    obj_pts = obj_points[b, o].cpu().numpy()  # (num_points, 3)
                    
                    # Sample and normalize point cloud (like preprocessing)
                    sampled_points = point_cloud.sample_and_normalize_pcl(obj_pts)
                    points_pt = torch.from_numpy(sampled_points).unsqueeze(0).to(self.device).float()
                    
                    # Extract features using I2PMAE
                    with torch.no_grad():
                        obj_feat = self.point_feature_extractor(points_pt)  # (1, feat_dim)
                    
                    batch_features.append(obj_feat.squeeze(0))
                else:
                    # Zero features for invalid objects
                    batch_features.append(torch.zeros(self.feat_dims[ModalityType.CAD]).to(self.device))
            
            object_features.append(torch.stack(batch_features))
        
        # Stack all batch features: (batch_size, num_objects, feat_dim)
        cad_features = torch.stack(object_features)
        
        # Project to output dimension
        cad_embeddings = self.modality_projections[ModalityType.CAD](cad_features.view(-1, cad_features.size(-1)))
        cad_embeddings = cad_embeddings.view(batch_size, num_objects, -1)
        return cad_embeddings
    
    def encode_rgb_objects(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Encode RGB images for objects"""
        # rgb_images: (batch_size, num_objects, num_views, C, H, W)
        batch_size, num_objects, num_views = rgb_images.size(0), rgb_images.size(1), rgb_images.size(2)
        
        # Flatten to process all object views at once
        rgb_flat = rgb_images.view(-1, rgb_images.size(-3), rgb_images.size(-2), rgb_images.size(-1))
        
        # Extract features using DINOv2
        rgb_embedding = self.encoder2D(rgb_flat)  # (batch*objects*views, num_patches, embed_dim)
        
        # Use only CLS token (like 2D preprocessing)
        rgb_embedding = rgb_embedding[:, 0, :]  # (batch*objects*views, embed_dim)
        
        # Reshape and aggregate views per object with simple mean (like preprocessing)
        rgb_embedding = rgb_embedding.view(batch_size, num_objects, num_views, -1)
        rgb_embedding = rgb_embedding.mean(dim=2)  # Average across views
        rgb_embedding = rgb_embedding.view(batch_size, num_objects, -1)
        
        # Project to output dimension
        rgb_embeddings = self.modality_projections[ModalityType.RGB](rgb_embedding.view(-1, rgb_embedding.size(-1)))
        rgb_embeddings = rgb_embeddings.view(batch_size, num_objects, -1)
        return rgb_embeddings
    
    def encode_referral_objects(self, referral_texts: List[List[List[str]]]) -> torch.Tensor:
        """Encode referral texts for objects"""
        # referral_texts: List[List[List[str]]] - batch_size x num_objects x referrals_per_object
        batch_embeddings = []
        
        for batch_texts in referral_texts:
            object_embeddings = []
            for obj_referrals in batch_texts:
                if not obj_referrals or (len(obj_referrals) == 1 and not obj_referrals[0].strip()):
                    # Empty referral list or single empty string
                    object_embeddings.append(torch.zeros(self.out_dim).to(self.device))
                    continue
                
                # Extract features for all referrals of this object exactly like feat1D preprocessing
                referral_feats = []
                for text in obj_referrals:
                    if not text.strip():  # Skip empty text
                        continue
                    
                    # Use the exact same approach as preprocessing
                    encoded_text = self.encoder1D.tokenizer(text, padding=True, add_special_tokens=True, return_tensors="pt").to(self.device)
                    if encoded_text['input_ids'].shape[1] > 512:
                        continue
                    
                    with torch.no_grad():
                        # Use text_encoder like preprocessing (self.model.text_encoder)
                        encoded_text = self.encoder1D.text_encoder(encoded_text.input_ids, attention_mask=encoded_text.attention_mask,
                                                                 return_dict=True, mode='text').last_hidden_state[:, 0]  # (1, 768)
                        referral_feats.append(encoded_text.cpu().numpy().reshape(1, -1))  # Convert to numpy like preprocessing
                
                if referral_feats:
                    # Take mean exactly like preprocessing: np.mean(object_referral_feats, axis=0).reshape(1, -1)
                    referral_feats = np.concatenate(referral_feats)  # Concatenate all features
                    mean_feats = np.mean(referral_feats, axis=0).reshape(1, -1)  # Take mean like preprocessing
                    
                    # Convert back to torch and project to output dimension
                    mean_embedding = torch.from_numpy(mean_feats.squeeze(0)).to(self.device).float()
                    projected_embedding = self.modality_projections[ModalityType.REF](mean_embedding.unsqueeze(0)).squeeze(0)
                    object_embeddings.append(projected_embedding)
                else:
                    # No valid referrals
                    object_embeddings.append(torch.zeros(self.out_dim).to(self.device))
            
            batch_embeddings.append(torch.stack(object_embeddings))
        
        return torch.stack(batch_embeddings)
    
    def encode_1d_from_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text directly to 1D features"""
        text_embedding = None
        for text in texts:
            encoded_text = self.tokenizer(text, padding=True, add_special_tokens=True, return_tensors="pt").to(self.device)
            if encoded_text['input_ids'].shape[1] > 512:
                continue
            
            encoded_text = self.encoder1D(encoded_text.input_ids, attention_mask=encoded_text.attention_mask,
                                         return_dict=True, mode='text').last_hidden_state[:, 0]
            
            text_embedding = encoded_text if text_embedding is None else torch.cat((text_embedding, encoded_text), dim=0)
        
        if text_embedding is not None:
            text_embedding = text_embedding[0, :].unsqueeze(0)
            text_embedding = self.modality_projections[ModalityType.REF](text_embedding)
        
        return text_embedding

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        embedding_dict = {'embeddings': {}, 'masks': {}}
        
        # Process each modality
        if 'objects' in data_dict and 'inputs' in data_dict['objects']:
            objects_inputs = data_dict['objects']['inputs']
            masks = data_dict.get('masks', {})
            
            # Point modality
            if 'point' in objects_inputs:
                point_mask = masks.get('point', torch.ones(objects_inputs['point'].size(0), objects_inputs['point'].size(1))).to(self.device)
                embedding_dict['embeddings']['point'] = self.encode_point_objects(
                    objects_inputs['point'].to(self.device),
                    point_mask
                )
                embedding_dict['masks']['point'] = point_mask
            
            # CAD modality
            if 'cad' in objects_inputs:
                cad_mask = masks.get('cad', torch.ones(objects_inputs['cad'].size(0), objects_inputs['cad'].size(1))).to(self.device)
                embedding_dict['embeddings']['cad'] = self.encode_cad_objects(
                    objects_inputs['cad'].to(self.device),
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