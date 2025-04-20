import os
import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
import json
import argparse
import misc.utils
BASE_PATH = "/Users/gauravpradeep/CrossOver_ScaleUp/Structured3D/scans/"


def create_color_palette():
    """Returns the NYU40 colormap mapping RGB to class indices."""
    return [
       (0, 0, 0),  # Unlabeled (0)
       (174, 199, 232),  # wall (1)
       (152, 223, 138),  # floor (2)
       (31, 119, 180),  # cabinet (3)
       (255, 187, 120),  # bed (4)
       (188, 189, 34),  # chair (5)
       (140, 86, 75),  # sofa (6)
       (255, 152, 150),  # table (7)
       (214, 39, 40),  # door (8)
       (197, 176, 213),  # window (9)
       (148, 103, 189),  # bookshelf (10)
       (196, 156, 148),  # picture (11)
       (23, 190, 207),  # counter (12)
       (178, 76, 76),  
       (247, 182, 210),  # desk (14)
       (66, 188, 102), 
       (219, 219, 141),  # curtain (16)
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),  # refrigerator (25)
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),  # shower curtain (28)
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  # toilet (33)
       (112, 128, 144),  # sink (34)
       (96, 207, 209), 
       (227, 119, 194),  # bathtub (36)
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  # otherfurn (39)
       (100, 85, 144)
    ]

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

def point_inside_bbox(point, bbox_corners):
    """Check if a point is inside a 3D bounding box defined by its 8 corners."""
    min_coords = np.min(bbox_corners, axis=0)
    max_coords = np.max(bbox_corners, axis=0)

    return np.all(min_coords <= point) and np.all(point <= max_coords)

def load_bounding_boxes(bbox_json_path):
    """Load 3D bounding boxes from a JSON file."""
    with open(bbox_json_path, 'r') as f:
        bboxes = json.load(f)
    return bboxes

def rgb_to_nyu40id(rgb_image):
    """Convert RGB values from `semantic.png` to corresponding NYU40 IDs."""
    palette = create_color_palette()
    color_to_id = {color: idx for idx, color in enumerate(palette)}

    h, w, _ = rgb_image.shape
    rgb_flatten = rgb_image.reshape(-1, 3)

    # Convert each RGB value to corresponding NYU40 ID
    nyu40_ids = np.array([color_to_id.get(tuple(rgb), 0) for rgb in rgb_flatten], dtype=np.int32)
    
    return nyu40_ids.reshape(h, w)


def save_ply_with_labels(filename, pointcloud, object_ids, nyu40_ids):
    """Save PLY file with object_id and nyu40id."""
    points = np.asarray(pointcloud.points)
    colors = (np.asarray(pointcloud.colors) * 255).astype(np.uint8) if pointcloud.has_colors() else np.zeros_like(points, dtype=np.uint8)

    vertex_data = np.array(
        list(zip(
            points[:, 0], points[:, 1], points[:, 2],  # x, y, z
            colors[:, 0], colors[:, 1], colors[:, 2],  # red, green, blue
            np.full(len(points), 255, dtype=np.uint8),  # alpha
            object_ids,  # Object ID
            nyu40_ids  # NYU40 Semantic ID
        )),
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'),
            ('object_id', 'i4'),
            ('nyu40id', 'i4')
        ]
    )

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=False).write(filename)
    
def process_room(scene_id, room_id, room_path):
    """Processes a single room by merging all views and generating a 3D mesh."""
    pcd_list = []
    object_ids_list = []
    nyu40_ids_list = []

    # Iterate over all views in the room
    for view_id in sorted(os.listdir(room_path)):
        view_path = os.path.join(room_path, view_id)

        rgb_image_path = os.path.join(view_path, "rgb_rawlight.png")
        depth_image_path = os.path.join(view_path, "depth.png")
        camera_path = os.path.join(view_path, "camera_pose.txt")
        # instance_image_path = os.path.join(view_path, "instance.png")
        semantic_image_path = os.path.join(view_path, "semantic.png")

        if not all(os.path.exists(p) for p in [rgb_image_path, depth_image_path, camera_path, semantic_image_path]):
            print(f"Skipping Scene {scene_id}, Room {room_id}, View {view_id}: Missing files")
            continue

        print(f"Processing Scene {scene_id}, Room {room_id}, View {view_id}...")

        color = cv2.imread(rgb_image_path)
        # cv2.imshow("color", color)
        # cv2.waitKey(0)
        # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # Convert mm to meters
        # instance = cv2.imread(instance_image_path, cv2.IMREAD_UNCHANGED)  # Object ID image
        semantic = cv2.imread(semantic_image_path)  # Read as BGR
        semantic = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)  # Convert to RGB

        nyu40_id_image = rgb_to_nyu40id(semantic)

        valid_mask = depth.flatten() > 0
        # object_ids = instance.flatten()[valid_mask]
        nyu40_ids = nyu40_id_image.flatten()[valid_mask]

        height, width = color.shape[:2]
        camera_info = np.loadtxt(camera_path)
        rot, trans, K = parse_camera_info(camera_info, height, width)
        trans = np.array(trans) / 1000
        
        
        color_o3d = o3d.geometry.Image(color)
        depth_o3d = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
        )
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot.T
        extrinsic[:3, -1] = trans
        extrinsic = np.linalg.inv(extrinsic)
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0][0], K[1][1], K[0][2], K[1][2])
        pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic)

        pcd_list.append(pointcloud)
        # object_ids_list.append(object_ids)
        nyu40_ids_list.append(nyu40_ids)

    if not pcd_list:
        print(f"Skipping Scene {scene_id}, Room {room_id}: No valid views.")
        return

    pcd_combined = pcd_list[0]
    for pcd in pcd_list[1:]:
        pcd_combined += pcd
    
    object_ids_combined = np.array([-1]*len(np.asarray(pcd_combined.points)), dtype=int)  # Initialize object IDs

    # Efficient assignment of object IDs based on bounding box inclusion
    points = np.asarray(pcd_combined.points)
    colors = np.asarray(pcd_combined.colors)
    
    
    bboxes_json_path = os.path.join(BASE_PATH, scene_id, "bbox_3d.json")
    bboxes = load_bounding_boxes(bboxes_json_path)
    for idx, bbox in enumerate(bboxes):
        basis = np.array(bbox['basis'])
        coeffs = np.array(bbox['coeffs'])
        centroid = np.array(bbox['centroid'])
        bbox_corners = misc.utils.get_corners_of_bb3d_no_index(basis, coeffs, centroid)  # 8 corners of the bounding box
        bbox_corners =  bbox_corners / 1000
        # Create mask for points inside this bounding box
        box_min = np.min(bbox_corners, axis=0, keepdims=True)
        box_max = np.max(bbox_corners, axis=0, keepdims=True)
        # print(min_corner, max_corner)
        # print(points)
        # mask = np.all((points >= box_min) & (points <= max_corner), axis=1)
        point_max_mask = np.all(points < box_max, axis=1)
        point_min_mask = np.all(points > box_min, axis=1)
        point_mask = np.logical_and(point_max_mask, point_min_mask)
        points_in_bbox = points[point_mask]
        # print(points_in_bbox.shape)
        # if points_in_bbox.shape[0] != 0:
        #     print(bbox['ID'])
        #     colors_in_bbox = colors[mask]
        #     object_pcd = o3d.geometry.PointCloud()
        #     object_pcd.points = o3d.utility.Vector3dVector(points_in_bbox)
        #     object_pcd.colors = o3d.utility.Vector3dVector(colors_in_bbox)
        #     o3d.visualization.draw_geometries([object_pcd])
        # print(np.all(points>=min_corner, axis=1))
        # Assign object ID to points inside this bounding box
        object_ids_combined[point_mask] = bbox['ID']
    # o3d.visualization.draw_geometries([pcd_combined])
   
    
    nyu40_ids_combined = np.concatenate(nyu40_ids_list)
    # print(np.unique(object_ids_combined))
    # Save the mesh file
    output_dir = os.path.join(BASE_PATH, scene_id, "3D_rendering", room_id)
    os.makedirs(output_dir, exist_ok=True)
    ply_filename = os.path.join(output_dir, "room_mesh.ply")

    save_ply_with_labels(ply_filename, pcd_combined, object_ids_combined, nyu40_ids_combined)
    print(f"Saved mesh for Scene {scene_id}, Room {room_id} -> {ply_filename}")


# if __name__ == '__main__':
#     for scene_id in sorted(os.listdir(BASE_PATH)):
#         scene_path = os.path.join(BASE_PATH, scene_id, "2D_rendering")
#         if not os.path.isdir(scene_path):
#             continue

#         for room_id in sorted(os.listdir(scene_path)):
#             room_path = os.path.join(scene_path, room_id, "perspective", "full")
#             if os.path.isdir(room_path):
                # process_room(scene_id, room_id, room_path)
def parse_args():
    parser = argparse.ArgumentParser(description='Generate PLY files from Structured3D dataset')
    parser.add_argument('--base_path', type=str, default="/Users/gauravpradeep/CrossOver_ScaleUp/Structured3D/scans/",
                        help='Base path to the Structured3D dataset')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    BASE_PATH = args.base_path
    
    for scene_id in sorted(os.listdir(BASE_PATH)):
        scene_path = os.path.join(BASE_PATH, scene_id, "2D_rendering")
        if not os.path.isdir(scene_path):
            continue

        for room_id in sorted(os.listdir(scene_path)):
            room_path = os.path.join(scene_path, room_id, "perspective", "full")
            if os.path.isdir(room_path):
                process_room(scene_id, room_id, room_path)
# ---------------------------------------
# instance image based object id assignment
# ---------------------------------------

# def process_room(scene_id, room_id, room_path):
#     """Processes a single room by merging all views and generating a 3D mesh."""
#     pcd_list = []
#     object_ids_list = []
#     nyu40_ids_list = []

#     # Iterate over all views in the room
#     for view_id in sorted(os.listdir(room_path)):
#         view_path = os.path.join(room_path, view_id)

#         rgb_image_path = os.path.join(view_path, "rgb_rawlight.png")
#         depth_image_path = os.path.join(view_path, "depth.png")
#         camera_path = os.path.join(view_path, "camera_pose.txt")
#         instance_image_path = os.path.join(view_path, "instance.png")
#         semantic_image_path = os.path.join(view_path, "semantic.png")

#         if not all(os.path.exists(p) for p in [rgb_image_path, depth_image_path, camera_path, instance_image_path, semantic_image_path]):
#             print(f"Skipping Scene {scene_id}, Room {room_id}, View {view_id}: Missing files")
#             continue

#         print(f"Processing Scene {scene_id}, Room {room_id}, View {view_id}...")

#         color = cv2.imread(rgb_image_path)
#         depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # Convert mm to meters
#         instance = cv2.imread(instance_image_path, cv2.IMREAD_UNCHANGED)  # Object ID image
#         semantic = cv2.imread(semantic_image_path)  # Read as BGR
#         semantic = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)  # Convert to RGB

#         nyu40_id_image = rgb_to_nyu40id(semantic)

#         valid_mask = depth.flatten() > 0
#         object_ids = instance.flatten()[valid_mask]
#         nyu40_ids = nyu40_id_image.flatten()[valid_mask]

#         height, width = color.shape[:2]
#         camera_info = np.loadtxt(camera_path)
#         rot, trans, K = parse_camera_info(camera_info, height, width)
#         trans = np.array(trans) / 1000
        
        
#         color_o3d = o3d.geometry.Image(color)
#         depth_o3d = o3d.geometry.Image(depth)
#         rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
#         )
#         extrinsic = np.eye(4)
#         extrinsic[:3, :3] = rot.T
#         extrinsic[:3, -1] = trans
#         extrinsic = np.linalg.inv(extrinsic)
        
#         intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0][0], K[1][1], K[0][2], K[1][2])
#         pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic)

#         pcd_list.append(pointcloud)
#         object_ids_list.append(object_ids)
#         nyu40_ids_list.append(nyu40_ids)

#     if not pcd_list:
#         print(f"Skipping Scene {scene_id}, Room {room_id}: No valid views.")
#         return

#     pcd_combined = pcd_list[0]
#     for pcd in pcd_list[1:]:
#         pcd_combined += pcd
#     # o3d.visualization.draw_geometries([pcd_combined])

#     object_ids_combined = np.concatenate(object_ids_list)
#     nyu40_ids_combined = np.concatenate(nyu40_ids_list)

#     # Save the mesh file
#     output_dir = os.path.join(BASE_PATH, scene_id, "3D_rendering", room_id)
#     os.makedirs(output_dir, exist_ok=True)
#     ply_filename = os.path.join(output_dir, "room_mesh.ply")

#     save_ply_with_labels(ply_filename, pcd_combined, object_ids_combined, nyu40_ids_combined)
#     print(f"Saved mesh for Scene {scene_id}, Room {room_id} -> {ply_filename}")

