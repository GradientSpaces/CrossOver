import os.path as osp
import torch
import numpy as np
from tqdm import tqdm

from common import load_utils 
from util import structured3d
from util.structured3d import S3D_SCANNET
from preprocess.build import PROCESSOR_REGISTRY
from preprocess.feat1D.base import Base1DProcessor


@PROCESSOR_REGISTRY.register()
class Structured3D_1DProcessor(Base1DProcessor):
    def __init__(self, config_data, config_1D, split) -> None:
        super(Structured3D_1DProcessor, self).__init__(config_data, config_1D, split)
        self.data_dir = config_data.base_dir
        
        files_dir = osp.join(config_data.base_dir, 'files')
        
        self.scan_ids = []
        self.scan_ids = structured3d.get_scan_ids(files_dir, split)
        
        self.out_dir = config_data.process_dir
        load_utils.ensure_dir(self.out_dir)        
        # Object Referrals
        self.object_referrals = load_utils.load_json(osp.join(files_dir, 'sceneverse/ssg_ref_rel2_template.json'))
        
    
    def compute1DFeaturesEachScan(self, scan_id):
        full_scan_id = scan_id
        scan_id = scan_id.split('_')
        room_id = scan_id[-1]
        scan_id = scan_id[0]+'_'+scan_id[1]
        obj2tgtid_map = load_utils.load_json(osp.join(self.data_dir,'scans',scan_id,'2D_rendering',room_id,'obj2tgid.json'))
        
        scene_out_dir = osp.join(self.out_dir, full_scan_id)
        load_utils.ensure_dir(scene_out_dir)
        
        objectID_to_labelID_map = torch.load(osp.join(scene_out_dir, 'object_id_to_label_id_map.pt'))['obj_id_to_label_id_map'] 
        
        object_referral_embeddings, scene_referral_embeddings = {}, None
        if len(objectID_to_labelID_map.keys()) != 0:
            object_referral_embeddings = self.computeObjectWise1DFeaturesEachScan(full_scan_id, objectID_to_labelID_map, obj2tgtid_map)

        scene_referrals = [referral for referral in self.object_referrals if referral['scan_id'] == full_scan_id]
        
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
             
    def computeObjectWise1DFeaturesEachScan(self, scan_id, objectID_to_labelID_map, obj2tgtid):
        object_referral_embeddings = {}
        matched_objids=[]
        scan_referrals = [referral for referral in self.object_referrals if referral['scan_id'] == scan_id]
        
        for instance_id in objectID_to_labelID_map.keys():
            if str(instance_id) not in obj2tgtid.keys():
                # print(f"Instance ID {instance_id} not found in obj2tgtid mapping for scan {scan_id}. Skipping...")
                continue
            mapped_obj_id = obj2tgtid[str(instance_id)]
            nyu40id= objectID_to_labelID_map[instance_id]
            if nyu40id==0:
                continue
            label = S3D_SCANNET[nyu40id]
            object_referral = []
            for referral in scan_referrals:
                if int(referral['target_id']) == int(mapped_obj_id):
                    if referral['instance_type'] == label:
                    # print(referral['utterance'])
                        matched_objids.append(instance_id)
                        # print(scan_id,label,referral['instance_type'],referral['target_id'],mapped_obj_id)
                        object_referral.append(referral['utterance'])
                    # else:
                        # print(scan_id,label,referral['instance_type'],referral['target_id'],mapped_obj_id)
                    
            if len(object_referral) != 0:
                # print(scan_id,instance_id,len(object_referral))
                object_referral_feats = self.extractTextFeats(object_referral)    
                if object_referral_feats is not None:
                    object_referral_feats = np.mean(object_referral_feats, axis = 0).reshape(1, -1)
                    assert object_referral_feats.shape == (1, self.embed_dim)
                    
                    object_referral_embeddings[instance_id] = {'referral' : object_referral, 'feats' : object_referral_feats}

        # finding unmatched referrals
        unmatched_referrals = []
        for referral in scan_referrals:
            mapped_obj_id = referral['target_id']
            if int(mapped_obj_id) not in [int(obj2tgtid[str(instance_id)]) for instance_id in objectID_to_labelID_map.keys() if str(instance_id) in obj2tgtid]:
                unmatched_referrals.append(referral)
            elif any(int(mapped_obj_id) == int(obj2tgtid[str(instance_id)]) and S3D_SCANNET[objectID_to_labelID_map[instance_id]] != referral['instance_type'] 
                     for instance_id in objectID_to_labelID_map.keys() if str(instance_id) in obj2tgtid and objectID_to_labelID_map[instance_id] != 0):
                unmatched_referrals.append(referral)

        label_to_instances = {}
        for instance_id, nyu40id in objectID_to_labelID_map.items():
            if nyu40id == 0:
                continue
            label = S3D_SCANNET[nyu40id]
            if label not in label_to_instances:
                label_to_instances[label] = []
            label_to_instances[label].append(instance_id)

        for referral in unmatched_referrals:
            instance_type = referral['instance_type']
            if instance_type in label_to_instances and len(label_to_instances[instance_type]) == 1:
                instance_id = label_to_instances[instance_type][0]
                if instance_id not in matched_objids:
                    # print(f"Matching unmatched referral to unique instance: {scan_id},{instance_id}, {instance_type}, {referral['target_id']}")
                    if instance_id not in object_referral_embeddings:
                        object_referral = [referral['utterance']]
                    else:
                        object_referral_embeddings[instance_id]['referral'].append(referral['utterance'])
                    
                    object_referral_feats = self.extractTextFeats(object_referral)
                    if object_referral_feats is not None:
                        object_referral_feats = np.mean(object_referral_feats, axis=0).reshape(1, -1)
                        object_referral_embeddings[instance_id] = {'referral': object_referral, 'feats': object_referral_feats}

    
        # print(object_referral_embeddings.keys())
        return object_referral_embeddings
