import os.path as osp
import numpy as np
from plyfile import PlyData
from glob import glob
import cv2

S3D_SCANNET = {
    1: 'wall',
    2: 'floor',
    3: 'cabinet',
    4: 'bed',
    5: 'chair',
    6: 'sofa',
    7: 'table',
    8: 'door',
    9: 'window',
    10: 'bookshelf',
    11: 'picture',
    12: 'counter',
    13: 'blinds',
    14: 'desk',
    15: 'shelf',
    16: 'curtain',
    17: 'dresser',
    18: 'pillow',
    19: 'mirror',
    20: 'mat',
    21: 'clothes',
    22: 'ceiling',
    23: 'books',
    24: 'refrigerator',
    25: 'tv',
    26: 'paper',
    27: 'towel',
    28: 'shower curtain',
    29: 'box',
    30: 'whiteboard',
    31: 'person',
    32: 'nightstand',
    33: 'toilet',
    34: 'sink',
    35: 'lamp',
    36: 'bathtub',
    37: 'bag',
    38: 'otherstructure',
    39: 'otherfurniture',
    40: 'otherprop'}

def get_scan_ids(dirname, split):
    filepath = osp.join(dirname, '{}_scans.txt'.format(split))
    scan_ids = np.genfromtxt(filepath, dtype = str)
    return scan_ids

def load_ply_data(data_dir, scan_id, room_id):
    
    filename_in = osp.join(data_dir, scan_id, '3D_rendering', room_id, 'room_mesh.ply')
    print(scan_id)
    if not osp.exists(filename_in):
        raise FileNotFoundError(f"PLY file not found: {filename_in}")
    
    with open(filename_in, 'rb') as file:
        ply_data = PlyData.read(file)
    
    x = np.array(ply_data['vertex']['x'])
    y = np.array(ply_data['vertex']['y'])
    z = np.array(ply_data['vertex']['z'])
    red = np.array(ply_data['vertex']['red'])
    green = np.array(ply_data['vertex']['green'])
    blue = np.array(ply_data['vertex']['blue'])
    vertex_object_ids = np.array(ply_data['vertex']['object_id']) 
    vertex_nyu40ids = np.array(ply_data['vertex']['nyu40id'])  
    # vertex_targetids = np.array(ply_data['vertex']['target_id'])  
    
    vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'),
            ('objectId', 'i4'),
            ('nyu40id', 'i4'),
            ('targetId', 'i4')
            
        ]
    
    scene_vertices = np.column_stack([x, y, z])    
    center_points = np.mean(scene_vertices, axis=0)
    center_points[2] = np.min(scene_vertices[:, 2])
    scene_vertices = scene_vertices - center_points
    
    vertices = np.empty(len(x), dtype=vertex_dtype)
    
    vertices['x'] = scene_vertices[:, 0].astype('f4')
    vertices['y'] = scene_vertices[:, 1].astype('f4')
    vertices['z'] = scene_vertices[:, 2].astype('f4')
    
    # vertices['x'] = x.astype('f4')
    # vertices['y'] = y.astype('f4')
    # vertices['z'] = z.astype('f4')
    
    vertices['red'] = red.astype('u1')
    vertices['green'] = green.astype('u1')
    vertices['blue'] = blue.astype('u1')
    vertices['objectId'] = vertex_object_ids.astype('i4')
    vertices['nyu40id'] = vertex_nyu40ids.astype('i4')
    vertices['targetId'] = np.zeros_like(x).astype('i4')
    # vertices['targetId'] = vertex_targetids.astype('i4')
    return vertices

def normalize(vector):
    return vector / np.linalg.norm(vector)
  

def parse_camera_info(camera_info, height, width):
    """ extract intrinsic and extrinsic matrix
    """
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = np.cross(W, up)
    V = np.cross(W, U)

    rot = np.vstack((U, V, W))

    trans = camera_info[:3]

    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    return rot, trans, K

def load_all_poses(scan_dir, frame_idxs):
    frame_poses = {}
    for frame_idx in frame_idxs:
        frame_pose = load_pose(scan_dir, frame_idx)
        frame_poses[frame_idx] = frame_pose
    return frame_poses

def load_pose(scan_dir, frame_id):
    pose_path = osp.join(scan_dir, frame_id, 'camera_pose.txt')
    camera_info = np.loadtxt(pose_path)
    rgb_image_path = osp.join(scan_dir, frame_id, 'rgb_rawlight.png')
    color = cv2.imread(rgb_image_path)
    height, width = color.shape[:2]
    rot, trans, K = parse_camera_info(camera_info, height, width)
    
    trans = np.array(trans) / 1000
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot.T
    extrinsic[:3, -1] = trans
    extrinsic = np.linalg.inv(extrinsic)    
    
    return extrinsic
        
def load_intrinsics(scene_folder):
    camera_info = np.loadtxt(osp.join(scene_folder, '0', 'camera_pose.txt'))
    rgb_image_path = osp.join(scene_folder, '0', 'rgb_rawlight.png')
    rgb_img = cv2.imread(rgb_image_path)
    height, width = rgb_img.shape[:2]
    _, _, K = parse_camera_info(camera_info, height, width)
    intrinsics = {}
    intrinsics['intrinsic_mat'] = K
    intrinsics['width'] = width
    intrinsics['height'] = height
    return intrinsics