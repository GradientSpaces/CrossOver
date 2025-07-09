import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import itertools

from fvcore.common.registry import Registry
import numpy as np

LOSS_REGISTRY = Registry("loss")

@LOSS_REGISTRY.register()
class RetrievalLoss(nn.Module):
    def __init__(self):
        super(RetrievalLoss, self).__init__()    
        self.logit_scale = nn.Parameter((torch.ones([]) * np.log(1 / 0.07)).exp())  
        
    def calculate_loss(self, src_embed: torch.tensor, ref_embed: torch.tensor, mask: torch.tensor=None) -> torch.tensor:
        logit_scale = torch.clamp(self.logit_scale, max=100)
        
        cross_logits_forward  = logit_scale * src_embed @ ref_embed.T
        cross_logits_backward = logit_scale * ref_embed @ src_embed.T
        
        labels = torch.arange(src_embed.shape[0], device=src_embed.device, dtype=torch.long)
        
        loss_forward  = F.cross_entropy(cross_logits_forward, labels, reduction='none')
        loss_backward = F.cross_entropy(cross_logits_backward, labels, reduction='none')
         
        loss = loss_forward + loss_backward
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def forward(self, data_dict: Dict[str, Any]) -> torch.tensor:
        loss_dict = {}
        base_type = 'object'
        a_embed = data_dict['embeddings'][base_type]
        
        for modality_type in data_dict['embeddings']:
            if modality_type == base_type: 
                continue
            
            b_embed = data_dict['embeddings'][modality_type]
            mask = torch.logical_and(data_dict['scene_masks'][base_type], data_dict['scene_masks'][modality_type]).reshape(-1, )
            
            if mask.sum() == 0:
                loss = torch.tensor(0.0, requires_grad=True, device=a_embed.device)
            else:
                loss = self.calculate_loss(a_embed, b_embed, mask)
            loss_dict[f'loss_{modality_type}'] = loss

        loss_dict['total_loss'] = sum(loss_dict.values())
        
        assert not torch.any(torch.isnan(loss_dict['total_loss'])), 'Loss Coming NaN!!!'
        
        return loss_dict['total_loss'], loss_dict

class ContrastiveLoss(nn.Module):
    def __init__(self, base_modality: str):
        super(ContrastiveLoss, self).__init__()      
        self.base_modality = base_modality
        self.logit_scale = nn.Parameter((torch.ones([]) * np.log(1 / 0.07)).exp())

    def calculate_loss(self, src_embed, ref_embed, mask):
        pass
    
    def forward(self, data_dict: Dict[str, Any]) -> torch.tensor:
        output_embeddings = data_dict['embeddings']
        loss_dict = {}
        
        for modality in output_embeddings.keys():
            if modality == self.base_modality: 
                continue
            
            mask = torch.logical_and(data_dict['masks'][self.base_modality], data_dict['masks'][modality])
            a_embed = output_embeddings[self.base_modality]
            b_embed = output_embeddings[modality]
            
            if mask.sum() == 0:
                loss = torch.tensor(0.0, requires_grad=True, device=a_embed.device)
            else:
                loss = self.calculate_loss(a_embed, b_embed, mask)
            
            loss_dict[f'loss_{modality}'] = loss
        
        total_loss = sum(loss_dict.values())
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

@LOSS_REGISTRY.register()
class ObjectWiseContrastiveLoss(ContrastiveLoss):
    def __init__(self, base_modality: str):
        super().__init__(base_modality)      

    def calculate_loss(self, src_embed: torch.Tensor, ref_embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logit_scale = torch.clamp(self.logit_scale, max=100)
        
        cross_logits_forward  = logit_scale * src_embed @ ref_embed.T
        cross_logits_backward = logit_scale * ref_embed @ src_embed.T
        
        labels = torch.arange(src_embed.shape[0], device=src_embed.device, dtype=torch.long)
        
        loss_forward  = F.cross_entropy(cross_logits_forward, labels, reduction='none')
        loss_backward = F.cross_entropy(cross_logits_backward, labels, reduction='none')
        
        loss = loss_forward + loss_backward
        loss = (loss * mask).sum() / mask.sum()
        return loss

@LOSS_REGISTRY.register()
class SceneWiseContrastiveLoss(ContrastiveLoss):
    def __init__(self, base_modality: str):
        super().__init__(base_modality)      

    def calculate_loss(self, src_embed: torch.Tensor, ref_embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logit_scale = torch.clamp(self.logit_scale, max=100)
        
        cross_logits_forward  = logit_scale * torch.matmul(src_embed, ref_embed.transpose(-1, -2))
        cross_logits_backward = logit_scale * torch.matmul(ref_embed, src_embed.transpose(-1, -2))
        
        # Labels for each scene
        B, N, _ = src_embed.shape
        labels = torch.arange(N, device=src_embed.device, dtype=torch.long).unsqueeze(0).repeat(B, 1)
        
        # Compute the cross entropy loss for both directions
        loss_forward  = F.cross_entropy(cross_logits_forward.permute(0, 2, 1), labels, reduction='none')
        loss_backward = F.cross_entropy(cross_logits_backward.permute(0, 2, 1), labels, reduction='none')
        
        loss = (loss_forward + loss_backward) / 2.0
        loss = (loss * mask).sum() / mask.sum()
        
        return loss

if __name__ == '__main__':
    data_dict = {'embeddings' : {}, 'masks' : {}}
    data_dict['embeddings']['point'] = torch.randn((5, 10, 768))
    data_dict['embeddings']['rgb']   = torch.randn((5, 10, 768))
    data_dict['masks']['point']   = torch.ones((5, 10))
    data_dict['masks']['rgb']   = torch.ones((5, 10))
    
    loss = SceneWiseContrastiveLoss(base_modality = 'rgb')
    total_loss, loss_dict = loss(data_dict)
    
