import numpy as np

class Annotations:
    """
    Annotation information
    """
    def __init__(self) -> None:
        self.gt_num = 0
        self.name = list()
        self.location = list()
        self.dimensions = list()
        self.gt_boxes_upright_depth = list()
        self.unaligned_location = list()
        self.unaligned_dimensions = list()
        self.unaligned_gt_boxes_upright_depth = list()
        self.index = list()
        self.classes = list()
        self.axis_align_matrix = list()

    def dump(self):
        """
        Dump information into dict
        """
        anno_dict = dict()
        anno_dict['gt_num'] = int(self.gt_num)
        anno_dict['name'] = np.asarray(self.name)
        anno_dict['location'] = np.asarray(self.location, dtype=np.float64)
        anno_dict['dimensions'] = np.asarray(self.dimensions, dtype=np.float64)
        anno_dict['gt_boxes_upright_depth'] = np.asarray(self.gt_boxes_upright_depth, \
            dtype=np.float64)
        anno_dict['unaligned_location'] = np.asarray(self.unaligned_location, \
            dtype=np.float64)
        anno_dict['unaligned_dimensions'] = np.asarray(self.unaligned_dimensions, \
            dtype=np.float64)
        anno_dict['unaligned_gt_boxes_upright_depth'] = np.asarray(
            self.unaligned_gt_boxes_upright_depth, dtype=np.float64)
        anno_dict['index'] = np.asarray(self.index, dtype=np.int32)
        anno_dict['class'] = np.asarray(self.classes, dtype=np.int64)
        anno_dict['axis_align_matrix'] = np.asarray(self.axis_align_matrix, dtype=np.float64)
        return anno_dict
    
    
class S3DUtilize(object):
    """
    Structured3D utilize functions
    """
    @staticmethod
    def get_fov_normal(image_size, cam_focal, norm=True):
        """
        Get the normal FoV directions
        """
        u2x, v2y = [(np.arange(1, image_size[a_i] + 1) - image_size[a_i] / 2) / cam_focal[a_i]\
            for a_i in [0, 1]]
        cam_m_u2x = np.tile([u2x], (image_size[1], 1))
        cam_m_v2y = np.tile(v2y[:, np.newaxis], (1, image_size[0]))
        cam_m_depth = np.ones(image_size).T
        fov_normal = np.stack((cam_m_depth, -1 * cam_m_v2y, cam_m_u2x), axis=-1)
        if norm:
            fov_normal = fov_normal / np.sqrt(np.sum(np.square(fov_normal), axis=-1, keepdims=True))
        return fov_normal

    @staticmethod
    def cast_perspective_to_local_coord(depth_img: np.ndarray, fov_normal):
        """
        Cast the perspective image into 3D coordinate system
        """
        return depth_img * fov_normal

    @staticmethod
    def cast_points_to_voxel(points, labels, room_size=(6.4, 3.2, 6.4), room_stride=0.2):
        """
        Voxelize the points
        """
        vol_resolution = (np.asarray(room_size) / room_stride).astype(np.int32)
        vol_index = np.floor(points / room_stride).astype(np.int32)
        in_vol = np.logical_and(np.all(vol_index < vol_resolution, axis=1), \
            np.all(vol_index >= 0, axis=1))
        v_x, v_y, v_z = [d_[..., 0] for d_ in np.split(vol_index[in_vol], 3, axis=-1)]
        vol_label = labels[in_vol]
        vol_data = np.zeros(vol_resolution, dtype=np.uint8)
        vol_data[v_x, v_y, v_z] = vol_label
        return vol_data

    @staticmethod
    def get_rotation_matrix_from_tu(cam_front, cam_up):
        """
        Get the rotation matrix from TU-coords
        """
        cam_n = np.cross(cam_front, cam_up)
        cam_m = np.stack((cam_front, cam_up, cam_n), axis=1).astype(np.float32)
        return cam_m

    @staticmethod
    def get_8points_bounding_box(basis, coeffs, centroid):
        """
        Get the 8 corners from the bounding box parameters
        """
        corners = np.zeros((8, 3))
        coeffs = np.abs(coeffs)
        corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * \
            coeffs[1] + basis[2, :] * coeffs[2]
        corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * \
            coeffs[1] + basis[2, :] * coeffs[2]
        corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * \
            coeffs[1] + basis[2, :] * coeffs[2]
        corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * \
            coeffs[1] + basis[2, :] * coeffs[2]

        corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * \
            coeffs[1] + -basis[2, :] * coeffs[2]
        corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * \
            coeffs[1] + -basis[2, :] * coeffs[2]
        corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * \
            coeffs[1] + -basis[2, :] * coeffs[2]
        corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * \
            coeffs[1] + -basis[2, :] * coeffs[2]
        corners = corners + np.tile(centroid, (8, 1))
        return corners
