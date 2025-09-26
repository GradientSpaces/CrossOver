import os.path as osp
import numpy as np
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as tvf
import torch

from common import load_utils
from util import scannet
from util import image as image_util
from common.load_utils import load_yaml
import albumentations as A

class ScannetInferDataset(Dataset):
    def __init__(self, data_dir, process_dir,
                 voxel_size=0.02, frame_skip=5, image_size=[224, 224]) -> None:
        self.voxel_size = voxel_size
        self.frame_skip = frame_skip
        self.image_size = image_size
        self.process_dir = process_dir
        self.scans_dir = osp.join(data_dir, 'scans')
        self.files_dir = osp.join(data_dir, 'files')
        self.floorplan_path = osp.join(process_dir, 'scans')
        self.referrals = load_utils.load_json(osp.join(self.files_dir, 'sceneverse/ssg_ref_rel2_template.json'))
        
        self.scan_ids = []
        for split in ['train', 'val']:
            filepath = osp.join(self.files_dir, 'scannetv2_{}.txt'.format(split))
            self.scan_ids.extend(np.genfromtxt(filepath, dtype = str))
        
        self.base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
        ])
        self.color_mean_std_path = osp.join(self.process_dir, 'color_mean_std.yaml')
        self.color_mean_std = load_yaml(self.color_mean_std_path)
        color_mean, color_std = (
            tuple(self.color_mean_std["mean"]),
            tuple(self.color_mean_std["std"]),
        )
        self.normalize_color = A.Normalize(mean=color_mean, std=color_std)
    
    def extract_images(self, scan_id, color_path):
        pose_data = scannet.load_poses(self.scans_dir, scan_id, skip=self.frame_skip)      
        frame_idxs = list(pose_data.keys())
        
        pose_data_arr = []
        for frame_idx in frame_idxs:
            pose = pose_data[frame_idx]
            rot_quat = R.from_matrix(pose[:3, :3]).as_quat()
            trans = pose[:3, 3]
            pose_data_arr.append([trans[0], trans[1], trans[2], rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]])
            
        pose_data_arr = np.array(pose_data_arr)
        sampled_frame_idxs = image_util.sample_camera_pos_on_grid(pose_data_arr)
        
        image_data = None
        for idx in sampled_frame_idxs:
            frame_index = frame_idxs[idx]
            image = Image.open(osp.join(color_path, '{}.jpg'.format(frame_index)))
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BICUBIC)
            image_pt = self.base_tf(image).unsqueeze(0)
            image_data = image_pt if image_data is None else torch.cat((image_data, image_pt), dim=0)

        return image_data.unsqueeze(0)
     
    def __getitem__(self, index):
        if isinstance(index, int):
            scan_id = self.scan_ids[index]
        
        if isinstance(index, str):
            scan_id = index
        
        scan_folder = osp.join(self.scans_dir, scan_id)
        data_dict = {}
        data_dict['masks'] = {}
        
        # Point Cloud
        mesh_file = osp.join(scan_folder, scan_id + '_vh_clean_2.ply')
        mesh_vertices = scannet.read_mesh_vertices_rgb(mesh_file)
        
        points = mesh_vertices[:, 0:3] 
        feats  = mesh_vertices[:, 3:]
        pseudo_image = feats.astype(np.uint8)[np.newaxis, :, :]
        feats = np.squeeze(self.normalize_color(image=pseudo_image)["image"])
        # feats /= 255.
        # feats -= 0.5
        
        _, sel = ME.utils.sparse_quantize(points / self.voxel_size, return_index=True)
        coords,  feats = points[sel], feats[sel]
        coords = np.floor(coords / self.voxel_size)
        coords-= coords.min(0)
        
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        data_dict['masks']['point'] = True
        
        # RGB
        color_path = osp.join(scan_folder, 'data/color')
        image_data = self.extract_images(scan_id, color_path)
        data_dict['masks']['rgb'] = True
        
        # Floorplan
        floorplan_imgpath = osp.join(self.floorplan_path, scan_id, 'floor+obj.png')
        if osp.exists(floorplan_imgpath):
            floorplan_img = Image.open(floorplan_imgpath)
            data_dict['masks']['floorplan'] = True
        else:
            floorplan_img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            floorplan_img = Image.fromarray(floorplan_img)
            data_dict['masks']['floorplan'] = False
    
        floorplan_img = floorplan_img.resize((self.image_size[1], self.image_size[0]), Image.BICUBIC)
        floorplan_data = self.base_tf(floorplan_img).unsqueeze(0)
        
        # Referral
        referrals = [referral for referral in self.referrals if referral['scan_id'] == scan_id]
        if len(referrals) != 0:
            if len(referrals) > 10:
                referrals = np.random.choice(referrals, size=10, replace=False)
            referrals = [referral['utterance'] for referral in referrals]
            referrals = [' '.join(referrals)]
            data_dict['masks']['referral'] = True
        else:
            referrals = ['']
            data_dict['masks']['referral'] = False
                
        data_dict['coordinates'] = coords
        data_dict['features'] = feats
        data_dict['rgb'] = image_data
        data_dict['floorplan'] = floorplan_data
        data_dict['referral'] = referrals
        
        return data_dict