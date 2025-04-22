import os.path as osp
import numpy as np
from plyfile import PlyData
from glob import glob
import csv
import json
def get_scan_ids(dirname: str, split: str) -> np.ndarray:
    """Retrieve scan IDs for the given directory and split."""
    filepath = osp.join(dirname, '{}_scans.txt'.format(split))
    scan_ids = np.genfromtxt(filepath, dtype = str)
    return scan_ids

def load_ply_data(data_dir, scan_id, label_file_name):
    with open(osp.join(data_dir, scan_id, 'mesh.refined.0.010000.segs.v2.json'), "r", encoding='utf-8') as f:
            segments = json.load(f)
    with open(osp.join(data_dir, scan_id, 'semseg.v2.json'), "r", encoding='utf-8') as f:
        aggregation = json.load(f)
            
    filename_in = osp.join(data_dir, scan_id, label_file_name)
    file = open(filename_in, 'rb')
    ply_data = PlyData.read(file)
    file.close()
    x = ply_data['vertex']['x']
    y = ply_data['vertex']['y']
    z = ply_data['vertex']['z']
    red = ply_data['vertex']['red']
    green = ply_data['vertex']['green']
    blue = ply_data['vertex']['blue']
    object_id = ply_data['vertex']['objectId']
    global_id = ply_data['vertex']['globalId']
    nyu40_id = ply_data['vertex']['NYU40']
    eigen13_id = ply_data['vertex']['Eigen13']
    rio27_id = ply_data['vertex']['RIO27']

    vertices = np.empty(len(x), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                                     ('objectId', 'h'), ('globalId', 'h'), ('NYU40', 'u1'), ('Eigen13', 'u1'), ('RIO27', 'u1')])
    
    seg_group = aggregation['segGroups']
    bbox_list = []
    for i, _ in enumerate(seg_group):
        rotation = np.array(seg_group[i]["obb"]["normalizedAxes"]).reshape(3, 3)
        transform = np.array(seg_group[i]["obb"]["centroid"]).reshape(-1, 3)
        scale = np.array(seg_group[i]["obb"]["axesLengths"]).reshape(-1, 3)
        trns = np.eye(4)
        trns[0:3, 3] = transform
        trns[0:3, 0:3] = rotation.T
        box3d = compute_box_3d(scale.reshape(3).tolist(), transform, rotation)
        bbox_list.append(box3d)
    
    align_angle = calc_align_matrix(bbox_list)
    scene_vertices = np.column_stack([x, y, z])    
    center_points = np.mean(scene_vertices, axis=0)
    center_points[2] = np.min(scene_vertices[:, 2])
    scene_vertices = scene_vertices - center_points
    
    scene_vertices = rotate_z_axis_by_degrees(np.array(scene_vertices), align_angle)
    
    vertices['x'] = scene_vertices[:, 0].astype('f4')
    vertices['y'] = scene_vertices[:, 1].astype('f4')
    vertices['z'] = scene_vertices[:, 2].astype('f4')
    # vertices['x'] = x.astype('f4')
    # vertices['y'] = y.astype('f4')
    # vertices['z'] = z.astype('f4')
    vertices['red'] = red.astype('u1')
    vertices['green'] = green.astype('u1')
    vertices['blue'] = blue.astype('u1')
    vertices['objectId'] = object_id.astype('h')
    vertices['globalId'] = global_id.astype('h')
    vertices['NYU40'] = nyu40_id.astype('u1')
    vertices['Eigen13'] = eigen13_id.astype('u1')
    vertices['RIO27'] = rio27_id.astype('u1')
    
    return vertices

def load_intrinsics(scan_dir: str, type: str = 'color') -> dict:
    """Load intrinsic information for the given scan directory and type."""
    info_path = osp.join(scan_dir, 'sequence', '_info.txt')

    width_search_string = 'm_colorWidth' if type == 'color' else 'm_depthWidth'
    height_search_string = 'm_colorHeight' if type == 'color' else 'm_depthHeight'
    calibration_search_string = 'm_calibrationColorIntrinsic' if type == 'color' else 'm_calibrationDepthIntrinsic'

    with open(info_path) as f:
        lines = f.readlines()
    
    for line in lines:
        if line.find(height_search_string) >= 0:
            intrinsic_height = line[line.find("= ") + 2 :]
        
        elif line.find(width_search_string) >= 0:
            intrinsic_width = line[line.find("= ") + 2 :]
        
        elif line.find(calibration_search_string) >= 0:
            intrinsic_mat = line[line.find("= ") + 2 :].split(" ")

            intrinsic_fx = intrinsic_mat[0]
            intrinsic_cx = intrinsic_mat[2]
            intrinsic_fy = intrinsic_mat[5]
            intrinsic_cy = intrinsic_mat[6]

            intrinsic_mat = np.array([[intrinsic_fx, 0, intrinsic_cx],
                                    [0, intrinsic_fy, intrinsic_cy],
                                    [0, 0, 1]])
            intrinsic_mat = intrinsic_mat.astype(np.float32)
    intrinsics = {'width' : float(intrinsic_width), 'height' : float(intrinsic_height), 
                  'intrinsic_mat' : intrinsic_mat}
    
    return intrinsics

def load_pose(scan_dir: str, frame_id: int) -> np.ndarray:
    """Load pose for a specific frame in the given scan directory."""
    pose_path = osp.join(scan_dir, 'sequence', 'frame-{}.pose.txt'.format(frame_id))
    pose = np.genfromtxt(pose_path)
    return pose

def load_all_poses(scan_dir: str, frame_idxs: list) -> dict:
    """Load all poses for specified frame indices in the scan directory."""
    frame_poses = {}
    for frame_idx in frame_idxs:
        frame_pose = load_pose(scan_dir, frame_idx)
        frame_poses[frame_idx] = frame_pose
    return frame_poses

def load_frame_idxs(scan_dir: str, skip: int = None) -> list:
    """Load frame indices from the scan directory, optionally skipping frames."""
    frames_paths = glob(osp.join(scan_dir, 'sequence', '*.jpg'))
    frame_names = [osp.basename(frame_path) for frame_path in frames_paths]
    frame_idxs = [frame_name.split('.')[0].split('-')[-1] for frame_name in frame_names]
    frame_idxs.sort()

    if skip is None:
        frame_idxs = frame_idxs
    else:
        frame_idxs = [frame_idx for frame_idx in frame_idxs[::skip]]
    return frame_idxs

def read_label_map(file_name: str, label_from: str = 'Global ID', label_to: str = 'Label') -> dict:
    """Read the label map from a CSV file mapping from one label to another."""
    assert osp.exists(file_name)
    
    raw_label_map = read_label_mapping(file_name, label_from=label_from, label_to=label_to)
    return raw_label_map

def read_label_mapping(filename: str, label_from: str = 'Global ID', label_to: str = 'Label') -> dict:
    """Read label mapping from a CSV file, converting keys to integers if applicable."""
    assert osp.isfile(filename)
    mapping = dict()
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            key = row[label_from].strip()  # Ensure any spaces are stripped
            value = row[label_to].strip()
            mapping[key] = value
    
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    
    return mapping

def represents_int(s: str) -> bool:
    """Check if the given string represents an integer."""
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
def calc_align_matrix(bbox_list):
    RANGE = [-45, 45]
    NUM_BIN = 90
    angles = np.linspace(RANGE[0], RANGE[1], NUM_BIN)
    angle_counts = {}
    for _a in angles:
        bucket = round(_a, 3)
        for box in bbox_list:
            box_r = rotate_z_axis_by_degrees(box, bucket)
            bottom = box_r[4:]
            if is_axis_aligned(bottom):
                angle_counts[bucket] = angle_counts.get(bucket, 0) + 1
    if len(angle_counts) == 0:
        RANGE = [-90, 90]
        NUM_BIN = 180
        angles = np.linspace(RANGE[0], RANGE[1], NUM_BIN)
        for _a in angles:
            bucket = round(_a, 3)
            for box in bbox_list:
                box_r = rotate_z_axis_by_degrees(box, bucket)
                bottom = box_r[4:]
                if is_axis_aligned(bottom, thres=0.15):
                    angle_counts[bucket] = angle_counts.get(bucket, 0) + 1
    most_common_angle = max(angle_counts, key=angle_counts.get)
    return most_common_angle

def is_axis_aligned(rotated_box, thres=0.05):
    x_diff = abs(rotated_box[0][0] - rotated_box[1][0])
    y_diff = abs(rotated_box[0][1] - rotated_box[3][1])
    return x_diff < thres and y_diff < thres

def rotate_z_axis_by_degrees(pointcloud, theta, clockwise=True):
    theta = np.deg2rad(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot_matrix = np.array([[cos_t, -sin_t, 0],
                           [sin_t, cos_t, 0],
                           [0, 0, 1]], pointcloud.dtype)
    if not clockwise:
        rot_matrix = rot_matrix.T
    return pointcloud.dot(rot_matrix)

def compute_box_3d(size, center, rotmat):
    """Compute corners of a single box from rotation matrix
    Args:
        size: list of float [dx, dy, dz]
        center: np.array [x, y, z]
        rotmat: np.array (3, 3)
    Returns:
        corners: (8, 3)
    """
    l, h, w = [i / 2 for i in size]
    center = np.reshape(center, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d = np.dot(
        np.transpose(rotmat), np.vstack([x_corners, y_corners, z_corners])
    )
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)