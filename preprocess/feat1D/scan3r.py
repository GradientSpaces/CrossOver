import os.path as osp
import torch
import numpy as np
from common import load_utils 
from util import scan3r
from typing import Dict, List, Union
import os
from preprocess.build import PROCESSOR_REGISTRY
from preprocess.feat1D.base import Base1DProcessor

@PROCESSOR_REGISTRY.register()
class Scan3R1DProcessor(Base1DProcessor):
    """Scan3R 1D feature (relationships) processor class."""
    def __init__(self, config_data, config_1D, split) -> None:
        super(Scan3R1DProcessor, self).__init__(config_data, config_1D, split)
        self.data_dir = config_data.base_dir
        
        files_dir = osp.join(config_data.base_dir, 'files')
        
        self.scan_ids = []
        self.scan_ids = scan3r.get_scan_ids(files_dir, split)
        
        self.out_dir = osp.join(config_data.process_dir, 'scans')
        load_utils.ensure_dir(self.out_dir)
        
        self.objects = load_utils.load_json(osp.join(files_dir, 'objects.json'))['scans']
        
        # Object Referrals
        self.object_referrals = load_utils.load_json(osp.join(files_dir, 'sceneverse/ssg_ref_rel2_template.json'))
        
        # label map
        self.undefined = 0
    
    def compute1DFeaturesEachScan(self, scan_id: str) -> None:
        data1D = {}
        scene_out_dir = osp.join(self.out_dir, scan_id)
        load_utils.ensure_dir(scene_out_dir)
        pt_1d_path = osp.join(scene_out_dir, "data1D.pt")
        if osp.exists(pt_1d_path):
            pt_data=torch.load(pt_1d_path)
            data1D['objects'] = pt_data['objects']
            data1D['scene'] = pt_data['scene']
            os.remove(pt_1d_path)
        else:
        # objectID_to_labelID_map = torch.load(osp.join(scene_out_dir, 'object_id_to_label_id_map.pt'))['obj_id_to_label_id_map']        
            npz_data = np.load(osp.join(scene_out_dir, 'object_id_to_label_id_map.npz'),allow_pickle=True)
            objectID_to_labelID_map = npz_data['obj_id_to_label_id_map'].item()
            scan_objects = [obj_data for obj_data in self.objects if obj_data['scan'] == scan_id][0]['objects']

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
            
            data1D['objects'] = {'referral_embeddings' : object_referral_embeddings}
            data1D['scene']   = {'referral_embedding': scene_referral_embeddings}
            
        # torch.save(data1D, osp.join(scene_out_dir, 'data1D.pt'))
        np.savez_compressed(osp.join(scene_out_dir, 'data1D.npz'), **data1D)
        
             
    def computeObjectWise1DFeaturesEachScan(self, scan_id: str, scan_objects: Dict, 
                                            objectID_to_labelID_map: Dict[int, int]) -> Dict[int, Dict[str, Union[List[str], np.ndarray]]]:
        object_referral_embeddings = {}
        
        scan_referrals = [referral for referral in self.object_referrals if referral['scan_id'] == scan_id]
        
        for idx, scan_object in enumerate(scan_objects):
            instance_id = int(scan_object['id'])
            
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