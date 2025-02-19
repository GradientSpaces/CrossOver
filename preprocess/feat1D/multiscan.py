import os.path as osp
import torch
import numpy as np
from tqdm import tqdm

from common import load_utils 
from util import labelmap, multiscan

from preprocess.build import PROCESSOR_REGISTRY
from preprocess.feat1D.base import Base1DProcessor

@PROCESSOR_REGISTRY.register()
class MultiScan1DProcessor(Base1DProcessor):
    def __init__(self, config_data, config_1D, split) -> None:
        super(MultiScan1DProcessor, self).__init__(config_data, config_1D, split)
        self.data_dir = config_data.base_dir
        
        files_dir = osp.join(config_data.base_dir, 'files')
        
        self.scan_ids = []
        self.scan_ids = multiscan.get_scan_ids(files_dir, split)
        
        self.out_dir = osp.join(config_data.process_dir, 'scans')
        load_utils.ensure_dir(self.out_dir)        
        # Object Referrals
        self.object_referrals = load_utils.load_json(osp.join(files_dir, 'sceneverse/ssg_ref_rel2_template.json'))
        
        # label map
        self.undefined = 0

    def load_objects_for_scan(self, scan_id):
        """Load and parse the annotations JSON for the given scan ID."""
        objects_path = osp.join(self.data_dir, 'scenes', scan_id, f"{scan_id}.annotations.json")
        if not osp.exists(objects_path):
            raise FileNotFoundError(f"Annotations file not found for scan ID: {scan_id}")
        
        annotations = load_utils.load_json(objects_path)
        objects = []
        
        for obj in annotations["objects"]:
            objects.append({
                "objectId": obj["objectId"],
                "global_id": obj.get("label")
            })
        
        return objects
    
    def extractTextFeats(self, texts, return_text = False):
        text_feats = []
        
        for text in texts:
            encoded_text = self.model.tokenizer(text, padding=True, add_special_tokens=True, return_tensors="pt").to(self.device)  
            if encoded_text['input_ids'].shape[1] > 512: 
                continue
            
            with torch.no_grad():
                encoded_text = self.model.text_encoder(encoded_text.input_ids, attention_mask = encoded_text.attention_mask,                      
                                                return_dict = True, mode = 'text').last_hidden_state[:, 0].cpu().numpy().reshape(1, -1)
                
            text_feats.append({'text' : text, 'feat' : encoded_text})
        
        if len(text_feats) == 0:
            return None
        
        if return_text:
            return text_feats
         
        text_feats = [text_feat['feat'] for text_feat in text_feats]
        text_feats = np.concatenate(text_feats)
        return text_feats
    
    
    def compute1DFeaturesEachScan(self, scan_id):
        scene_out_dir = osp.join(self.out_dir, scan_id)
        load_utils.ensure_dir(scene_out_dir)
        
        objectID_to_labelID_map = torch.load(osp.join(scene_out_dir, 'object_id_to_label_id_map.pt'))['obj_id_to_label_id_map']        
        scan_objects = self.load_objects_for_scan(scan_id)

        object_referral_embeddings, scene_referral_embeddings = {}, None
        if len(scan_objects) != 0:
            object_referral_embeddings = self.computeObjectWise1DFeaturesEachScan(scan_id, scan_objects, objectID_to_labelID_map)

        scene_referrals = [referral for referral in self.object_referrals if referral['scan_id'] == scan_id]
        
        if len(scene_referrals) != 0:
            if len(scene_referrals) > 10:
                scene_referrals = np.random.choice(scene_referrals, size=10, replace=False)
            
            scene_referrals = [scene_referral['utterance'] for scene_referral in scene_referrals]
            scene_referrals = ' '.join(scene_referrals)
            scene_referral_embeddings = self.extractTextFeats([scene_referrals], return_text=True)            
            assert scene_referral_embeddings is not None
        
        data1D = {}
        data1D['objects'] = {'referral_embeddings' : object_referral_embeddings}
        data1D['scene']   = {'referral_embedding': scene_referral_embeddings}
        
        torch.save(data1D, osp.join(scene_out_dir, 'data1D.pt'))
             
    def computeObjectWise1DFeaturesEachScan(self, scan_id, scan_objects, objectID_to_labelID_map):
        object_referral_embeddings = {}
        
        scan_referrals = [referral for referral in self.object_referrals if referral['scan_id'] == scan_id]
        
        for idx, scan_object in enumerate(scan_objects):
            instance_id = int(scan_object['objectId'])
            
            if instance_id not in objectID_to_labelID_map.keys():
                continue
            
            # Object Referral
            object_referral = [referral['utterance'] for referral in scan_referrals if int(referral['target_id']) == instance_id]
            if len(object_referral) != 0:
                object_referral_feats = self.extractTextFeats(object_referral)    
                if object_referral_feats is not None:
                    object_referral_feats = np.mean(object_referral_feats, axis = 0).reshape(1, -1)
                    assert object_referral_feats.shape == (1, self.embed_dim)
                    
                    object_referral_embeddings[instance_id] = {'referral' : object_referral, 'feats' : object_referral_feats}

            
        return object_referral_embeddings