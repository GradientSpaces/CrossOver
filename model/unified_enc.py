import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, List
from model.scenelevel_enc import SceneLevelEncoder
from modules.layers.patch_encoder import Mlps
from modules.layers.sparse_conv_encoder import ResNet34
from modules.basic_modules import get_mlp_head, MultiModalFusion
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class UnifiedEncoder(nn.Module):
    def __init__(self, args: DictConfig, modalities: List[str]) -> None:
        super().__init__()

        self.modalities = modalities
        self.out_dim = args.out_dim
        self.objectwise_modality_encoder = SceneLevelEncoder(args, self.modalities)
        
        self.objectwise_mlp_head = get_mlp_head(args.out_dim, args.out_dim, args.out_dim)
        self.fusion = MultiModalFusion(modal_num=len(self.modalities), with_weight=1)
        
        self.encoder3D = ResNet34(3, D=3)
        self.encoder3D_mlp_head = get_mlp_head(args.encoder3D.input_dim, args.encoder3D.input_dim, self.out_dim)
        
        self.encoder1D_mlp_head = get_mlp_head(args.encoder1D.input_dim, args.encoder1D.input_dim, self.out_dim)
        
        self.frame_mlp = Mlps(args.encoder2D.input_dim * 2, [args.encoder2D.input_dim], args.encoder2D.input_dim)
        self.encoder2D_mlp_head = get_mlp_head(args.encoder2D.input_dim, args.encoder2D.input_dim // 2, self.out_dim)        

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            data_dict = self.objectwise_modality_encoder(data_dict)
        
        data_dict['object_modality_embeddings'] = data_dict['embeddings']
        data_dict['embeddings'] = {}
        
        # x - (5, 16, 80, 768) -> (16, 80, 5, 768)
        object_modality_embed = [data_dict['object_modality_embeddings'][modality_key] for modality_key in data_dict['object_modality_embeddings']]
        object_modality_embed = torch.stack(object_modality_embed, dim=0) # modality x batch size x num_objects x embedding
        object_modality_embed = torch.permute(object_modality_embed, (1, 0, 2, 3)) # batch size x modality x num_objects x embedding
        
        mask = [data_dict['masks'][modality_key] for modality_key in data_dict['object_modality_embeddings']]
        mask = torch.stack(mask, dim=0) # modality x batch size x num objects
        mask = torch.permute(mask, (1, 0, 2)) # batch size x modality x num objects
        object_embeds = object_modality_embed * mask.unsqueeze(-1)
        
        valid_counts = mask.sum(dim=2) 
        object_embeds = object_embeds.sum(dim=2)
        
        valid_counts = valid_counts.unsqueeze(-1)  # Shape: (B, M, 1) for broadcasting during division
        object_embeds = object_embeds / valid_counts.clamp(min=1e-8)  # Shape: (B, M, D)
        assert not torch.any(torch.isnan(object_embeds)), 'Object Feats Coming NaN!!!'
        
        object_feats = self.fusion(object_embeds) # embedding size
        object_feats = self.objectwise_mlp_head(object_feats)
        
        data_dict['embeddings']['object']  = object_feats
        
        scene_feats_rgb = self.frame_mlp(data_dict['rgb_embedding'])
        scene_feats_rgb = scene_feats_rgb.mean(dim=1)
        scene_feats_rgb = self.encoder2D_mlp_head(scene_feats_rgb)
        data_dict['embeddings']['rgb'] = scene_feats_rgb
        
        scene_feats_pc = self.encoder3D(data_dict['pcl_sparse'])
        scene_feats_pc = self.encoder3D_mlp_head(scene_feats_pc)
        data_dict['embeddings']['point'] = scene_feats_pc
        
        scene_feats_floorplan = self.encoder2D_mlp_head(data_dict['floorplan_embedding'])
        data_dict['embeddings']['floorplan'] = scene_feats_floorplan
        
        scene_feats_referral = self.encoder1D_mlp_head(data_dict['referral_embedding'].to(torch.float32))
        data_dict['embeddings']['referral'] = scene_feats_referral
        
        assert not torch.any(torch.isnan(scene_feats_pc)), 'Scene Feats Coming NaN!!!'
        return data_dict
        
    def get_opt_params(self, lr: float) -> List[torch.nn.Parameter]:
        optimizer_grouped_parameters = []
        
        optimizer_grouped_parameters += self.encoder3D.parameters()
        optimizer_grouped_parameters += self.encoder3D_mlp_head.parameters()
        
        optimizer_grouped_parameters += self.objectwise_mlp_head.parameters()
        optimizer_grouped_parameters += self.encoder1D_mlp_head.parameters()
        
        optimizer_grouped_parameters += self.encoder2D_mlp_head.parameters()
        optimizer_grouped_parameters += self.frame_mlp.parameters()
        
        optimizer_grouped_parameters += self.fusion.parameters()
        
        for param in self.objectwise_modality_encoder.parameters():
            param.requires_grad = False
        
        return optimizer_grouped_parameters