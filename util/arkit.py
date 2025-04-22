import os.path as osp
import numpy as np
from plyfile import PlyData
from glob import glob
import csv
import jsonlines
import json
import os
import trimesh
import pandas as pd
import cv2

ARKITSCENE_SCANNET= {
'bed': 'bed',
'cabinet': 'cabinet',
'refrigerator': 'refrigerator',
'table': 'table',
'chair': 'chair',
'sink': 'sink',
'stove': 'stove',
'oven': 'oven',
'washer': 'washing machine',
'shelf': 'shelf',
'tv_monitor': 'tv',
'bathtub': 'bathtub',
'toilet': 'toilet',
'sofa': 'sofa',
'stool': 'stool',
'fireplace': 'fireplace',
'build_in_cabinet': 'cabinet',
'dishwasher': 'dishwasher',
'stairs': 'stairs'
}

def get_scan_ids(dirname, split):
    filepath = osp.join(dirname, '{}_scans.txt'.format(split))
    scan_ids = np.genfromtxt(filepath, dtype = str)
    return scan_ids

def load_frame_idxs(scan_dir, skip=None):
    frames_paths = glob(osp.join(scan_dir, f"{scan_dir.split('/')[-1]}_frames", 'lowres_wide', '*.png'))
    frame_names = [osp.basename(frame_path) for frame_path in frames_paths]
    frame_idxs = [frame_name.split('.png')[0].split("_")[1] for frame_name in frame_names]
    frame_idxs.sort() 

    if skip is not None:
        frame_idxs = frame_idxs[::skip]

    return frame_idxs

def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return Rt

def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def load_poses(scan_dir, scan_id, skip=None):
    frame_poses = {}
    frame_idxs = load_frame_idxs(scan_dir, skip=skip)
    traj_file = osp.join(scan_dir, f'{scan_id}_frames', 'lowres_wide.traj')
    with open(traj_file) as f:
            traj = f.readlines()
    for i,line in enumerate(traj):
        ts=line.split(" ")[0]
        rounded_ts = round(float(ts), 3)
        formatted_ts = f"{rounded_ts:.3f}"
        if formatted_ts not in frame_idxs:
            if f"{rounded_ts - 0.001:.3f}" in frame_idxs:
                frame_poses[f"{rounded_ts - 0.001:.3f}"] = TrajStringToMatrix(line)
            elif f"{rounded_ts + 0.001:.3f}" in frame_idxs:
                frame_poses[f"{rounded_ts + 0.001:.3f}"] = TrajStringToMatrix(line)
            else:
                print("no matching pose for frame", formatted_ts)
                continue
        # if f"{round(float(ts), 3):.3f}" not in frame_idxs:
        #     if f"{round(float(ts), 3)-0.001 :.3f}" in frame_idxs:
        #         frame_poses[f"{round(float(ts), 3)-0.001:.3f}"] = TrajStringToMatrix(line)
        #     elif f"{round(float(ts), 3)+0.001 :.3f}" in frame_idxs:
        #         frame_poses[f"{round(float(ts), 3)+0.001:.3f}"] = TrajStringToMatrix(line)
        #     else:    
        #         continue
        else:
            frame_poses[f"{round(float(ts), 3):.3f}"] = TrajStringToMatrix(line)
    # data = pd.read_csv(osp.join(scan_dir,f'{scan_id}_frames','lowres_wide.traj'), delim_whitespace=True, header=None)
    # for frame_idx,(index, row) in zip(frame_idxs,data.iterrows()):
    #     if skip is not None and index % skip != 0:
    #         continue
    #     rotation_axis = row[1:4].values
    #     rotation_angle = np.linalg.norm(rotation_axis)
    #     if rotation_angle != 0:
    #         rotation_axis = rotation_axis / rotation_angle
    #     translation = row[4:7].values
    #     # Convert axis-angle to rotation matrix
    #     # rotation_matrix = axis_angle_to_rotation_matrix(rotation_axis, rotation_angle)
    #     rotation_matrix=
    #     # Construct the 4x4 homogeneous transformation matrix
    #     homogenous_matrix = np.eye(4)
    #     homogenous_matrix[:3, :3] = rotation_matrix
    #     homogenous_matrix[:3, 3] = translation
    #     frame_poses[frame_idx] = homogenous_matrix
        
    return frame_poses

def axis_angle_to_rotation_matrix(axis, angle):
    # Normalize the rotation axis
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    # Compute the rotation matrix using the axis-angle formula
    rotation_matrix = np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])

    return rotation_matrix

def load_intrinsics(data_dir, scan_id, frame_id):
    '''
    Load ARKit intrinsic information
    '''
    pincam_path = osp.join(data_dir, scan_id, f'{scan_id}_frames', 'lowres_wide_intrinsics', f'{scan_id}_{frame_id}.pincam')
    if not os.path.exists(pincam_path):
        pincam_path = osp.join(data_dir, scan_id, f'{scan_id}_frames', 'lowres_wide_intrinsics', f'{scan_id}_{float(frame_id)-0.001:.3f}.pincam')
    if not os.path.exists(pincam_path):
        pincam_path = osp.join(data_dir, scan_id, f'{scan_id}_frames', 'lowres_wide_intrinsics', f'{scan_id}_{float(frame_id)+0.001:.3f}.pincam')
        
        
    intrinsics = {}

    # Read the .pincam file
    with open(pincam_path, "r") as f:
        line = f.readline().strip()
    
    # Parse the intrinsic parameters
    width, height, focal_length_x, focal_length_y, principal_point_x, principal_point_y = map(float, line.split())

    # Store the width and height
    intrinsics['width'] = width
    intrinsics['height'] = height

    # Construct the intrinsic matrix
    intrinsic_mat = np.array([
        [focal_length_x, 0, principal_point_x],
        [0, focal_length_y, principal_point_y],
        [0, 0, 1]
    ])
    intrinsics['intrinsic_mat'] = intrinsic_mat

    return intrinsics

def read_label_map(metadata_dir, label_from='raw_category', label_to='nyu40id'):
    LABEL_MAP_FILE = osp.join(metadata_dir, 'scannetv2-labels.combined.tsv')
    assert osp.exists(LABEL_MAP_FILE)
    
    raw_label_map = read_label_mapping(LABEL_MAP_FILE, label_from=label_from, label_to=label_to)
    return raw_label_map

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert osp.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = row[label_to]
    
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    
    return mapping

def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
def load_ply_data(data_dir, scan_id, annotations):
    filename_in = osp.join(data_dir, scan_id, f'{scan_id}_3dod_mesh.ply')
    file = open(filename_in, 'rb')
    plydata = PlyData.read(file)
    file.close()
    # plydata = trimesh.load(filename_in, process=False)
    vertices = plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']
    # vertices=plydata.vertices
    vertices = np.vstack(vertices).T

    vertex_colors = plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']
    vertex_colors = np.vstack(vertex_colors).T
    # vertex_colors = plydata.visual.vertex_colors
    # print("vertex_colors", vertex_colors.shape)
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                    ('objectId', 'h')]  
    vertices_structured = np.empty(vertices.shape[0], dtype=vertex_dtype)

    # Assign x, y, z, and color values to the structured array
    vertices_structured['red'] = vertex_colors[:, 0]
    vertices_structured['green'] = vertex_colors[:, 1]
    vertices_structured['blue'] = vertex_colors[:, 2]

    vertex_instance = np.zeros(vertices.shape[0], dtype='h')  # Use 'h' for signed 16-bit integer
    bbox_list=[]
    for _i, label_info in enumerate(annotations["data"]):
        object_id = _i + 1
        rotation = np.array(label_info["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3)

        transform = np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
        scale = np.array(label_info["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)

        trns = np.eye(4)
        trns[0:3, 3] = transform
        trns[0:3, 0:3] = rotation.T

        box_trimesh_fmt = trimesh.creation.box(scale.reshape(3,), trns)
        obj_containment = np.argwhere(box_trimesh_fmt.contains(vertices))

        vertex_instance[obj_containment] = object_id
        box3d = compute_box_3d(scale.reshape(3).tolist(), transform, rotation)
        bbox_list.append(box3d)
    
    # if len(bbox_list) == 0:
    #         return

    vertices_structured['objectId'] = vertex_instance

    align_angle = calc_align_matrix(bbox_list)

    vertices_aligned = rotate_z_axis_by_degrees(np.array(vertices), align_angle)

    if np.max(vertex_colors) <= 1:
        vertex_colors = vertex_colors * 255.0

    center_points = np.mean(vertices_aligned, axis=0)
    center_points[2] = np.min(vertices_aligned[:, 2]) 
    vertices_aligned = vertices_aligned - center_points

    vertices_structured['x'] = vertices_aligned[:, 0]
    vertices_structured['y'] = vertices_aligned[:, 1]
    vertices_structured['z'] = vertices_aligned[:, 2]
    
    # vertices_structured['x'] = plydata['vertex']['x']
    # vertices_structured['y'] = plydata['vertex']['y']
    # vertices_structured['z'] = plydata['vertex']['z']
    
    return vertices_structured

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
