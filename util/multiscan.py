import os.path as osp
import numpy as np
from plyfile import PlyData
from glob import glob
import csv
import jsonlines
import json
import os

MULTISCAN_SCANNET = {
    "wall": "wall",
    "door": "door",
    "slippers": "shoe",
    "mop": "broom",
    "rug": "rug",
    "floor": "floor",
    "basin": "sink",
    "basin_stand": "sink",
    "bucket": "bucket",
    "shower": "shower",
    "water_tank": "container",
    "beam": "wood beam",
    "pillar": "pillar",
    "ceiling": "ceiling",
    "sink": "sink",
    "toilet": "toilet",
    "cabinet": "cabinet",
    "remove": "object",
    "towel": "towel",
    "pillow": "pillow",
    "sofa": "sofa",
    "footstool": "footstool",
    "picture": "picture",
    "window": "window",
    "heater": "heater",
    "mirror": "mirror",
    "pipe": "pipe",
    "scarf": "cloth",
    "ceiling_light": "ceiling light",
    "chair": "chair",
    "table": "table",
    "vent": "vent",
    "bag": "bag",
    "wall_cabinet": "cabinet",
    "range": "stove",
    "ricemaker": "rice cooker",
    "pan": "cooking pan",
    "coffee_machine": "coffee maker",
    "rice_bag": "bag",
    "light": "light",
    "trashbin": "trash bin",
    "kettle": "kettle",
    "refrigerator": "refrigerator",
    "microwave": "microwave",
    "light_switch": "light switch",
    "rice_cooker": "rice cooker",
    "box": "box",
    "shoe": "shoe",
    "range_hood": "range hood",
    "wok": "cooking pan",
    "router": "object",
    "paper_towel": "paper towel roll",
    "stock_pot": "pot",
    "cutting_board": "cutting board",
    "wall_calendar": "calendar",
    "baseboard": "object",
    "coke_box": "box",
    "printer": "printer",
    "bowl": "bowl",
    "backpack": "backpack",
    "baseboard_heater": "heater",
    "broom": "broom",
    "dust_pan": "dustpan",
    "trash_bin": "trash bin",
    "rigid_duct": "vent",
    "electric_range": "stove",
    "spatula": "object",
    "faucet": "faucet",
    "bottle": "bottle",
    "countertop": "counter",
    "railing": "railing",
    "suitcase": "suitcase",
    "trash": "trash can",
    "pot": "pot",
    "kitchen_tool": "object",
    "vegetable": "object",
    "board": "board",
    "washing_machine": "washing machine",
    "jar": "jar",
    "object": "object",
    "notebook": "book",
    "induction_cooker": "stove",
    "instant_pot_lid": "cooking pot",
    "oven": "oven",
    "air_fryer": "object",
    "lid": "pot",
    "sponge": "sponge",
    "blender": "object",
    "spoon": "object",
    "dishwasher": "dishwasher",
    "detergent": "laundry detergent",
    "watermelon": "bananas",
    "yard_waste_bag": "garbage bag",
    "container": "container",
    "newspapers": "paper",
    "rag": "cloth",
    "ladder": "ladder",
    "gate": "door",
    "napkin_box": "tissue box",
    "jacket": "jacket",
    "windowsill": "windowsill",
    "water_faucet": "faucet",
    "steel_ball": "ball",
    "rice_maker": "rice cooker",
    "watter_bottle": "water bottle",
    "plastic_bag": "bag",
    "paper_bag": "paper bag",
    "cuttting_board": "cutting board",
    "trash_bin_lid": "trash bin",
    "hair_dryer": "hair dryer",
    "electric_socket": "power outlet",
    "electric_panel": "electric panel",
    "wash_stand": "sink",
    "soap": "soap",
    "curtain": "curtain",
    "bathtub": "bathtub",
    "smoke_detector": "smoke detector",
    "roll_paper": "paper towel roll",
    "chandelier": "chandelier",
    "hand_sanitizer": "hand sanitzer dispenser",
    "plate": "plate",
    "sticker": "sticker",
    "power_socket": "power outlet",
    "stacked_cups": "stack of cups",
    "stacked_chairs": "stack of chairs",
    "air_vent": "vent",
    "cornice": "cabinet",
    "wine_cabinet": "kitchen cabinet",
    "crock": "bowl",
    "liquor_box": "cabinet",
    "shampoo": "shampoo",
    "shower_curtain": "shower curtain",
    "wall_light": "wall lamp",
    "sink_cabinet": "sink",
    "toilet_roll": "toilet paper",
    "shelf": "shelf",
    "paper_bin": "recycling bin",
    "toilet_brush": "toilet brush",
    "shower_head": "shower head",
    "tv": "tv",
    "remote_control": "remote",
    "tv_box": "tv stand",
    "nightstand": "nightstand",
    "bed": "bed",
    "quilt": "blanket",
    "telephone": "telephone",
    "monitor": "monitor",
    "desk": "desk",
    "radiator_shell": "radiator",
    "calendar": "calendar",
    "clock": "clock",
    "keyboard": "keyboard",
    "speaker": "speaker",
    "clothes": "clothes",
    "door_frame": "doorframe",
    "sliding_door": "sliding door",
    "ceiling_lamp": "ceiling lamp",
    "scale": "scale",
    "power_strip": "power strip",
    "switch": "light switch",
    "basket": "basket",
    "stool": "stool",
    "shoes": "shoe",
    "slipper": "slippers",
    "bifold_door": "door",
    "rangehood": "range hood",
    "books": "books",
    "toilet_paper": "toilet paper",
    "mouse_pad": "mouse",
    "ipad": "ipad",
    "scissor": "knife block",
    "radiator": "radiator",
    "pc": "computer tower",
    "bicycle": "bicycle",
    "wardrobe": "wardrobe",
    "mouse": "mouse",
    "advertising_board": "poster",
    "banner": "banner",
    "ceiling_decoration": "ceiling light",
    "whiteboard": "whiteboard",
    "wall_storage_set": "shelf",
    "traffic_cone": "traffic cone",
    "wall_decoration": "decoration",
    "papers": "papers",
    "hat": "hat",
    "velvet_hangers": "clothes hanger",
    "circular_plate": "plate",
    "cellphone": "telephone",
    "pen": "keyboard piano",
    "paper": "paper",
    "lamp": "lamp",
    "curtain_box": "curtains",
    "woodcarving": "wood",
    "scissors": "knife block",
    "hand_dryer": "hand dryer",
    "machine": "machine",
    "vase": "vase",
    "plant": "plant",
    "power_socket_case": "power outlet",
    "gloves": "clothes",
    "dishcloth": "cloth",
    "painting": "painting",
    "shower_wall": "shower wall",
    "showerhead": "shower head",
    "tooth_mug": "cup",
    "map": "map",
    "knot_artwork": "decoration",
    "fan": "fan",
    "sphygmomanometer": "scale",
    "electric_kettle": "kettle",
    "bread_maker": "oven",
    "knife_set": "knife block",
    "soup_pot": "cooking pot",
    "flatware_set": "cutting board",
    "candle": "candle",
    "lid_rack": "dish rack",
    "flower": "flowerpot",
    "can": "can",
    "scoop": "bowl",
    "laptop": "laptop",
    "glass": "glass doors",
    "wet_floor_sign": "wet floor sign",
    "shower_enclosure": "shower doors",
    "jewelry_box": "jewelry box",
    "bath_brush": "hair brush",
    "sofa_cushion": "couch cushions",
    "tv_cabinet": "tv stand",
    "wood_fence": "wood beam",
    "floor_lamp": "lamp",
    "computer_case": "computer tower",
    "waste_container": "trash bin",
    "roadblock": "barricade",
    "trash_can_lids": "trash can",
    "hand_sanitizer_stand": "soap dispenser",
    "air_conditioner": "conditioner bottle",
    "pattern": "rug",
    "remote_controller": "remote",
    "phone": "telephone",
    "speakers": "speaker",
    "table_divider": "divider",
    "table_card": "card",
    "paper_trimmer": "paper cutter",
    "stapler": "stapler",
    "cup": "cup",
    "bathroom_heater": "heater",
    "wall_shelf": "shelf",
    "towel_rack": "towel",
    "sink_drain": "sink",
    "floor_drain": "floor",
    "broom_head": "broom",
    "door_curtain": "curtain",
    "refill_pouch": "plastic container",
    "bin": "bin",
    "stall_wall": "bathroom stall door",
    "wall_speaker": "speaker",
    "laundry_basket": "laundry basket",
    "tissue_box": "tissue box",
    "document_holder": "file cabinet",
    "yoga_mat": "yoga mat",
    "gas_range": "stove",
    "chopping_board": "cutting board",
    "book_scanner": "scanner",
    "payment_terminal": "vending machine",
    "napkin_roll": "paper towel roll",
    "faucet_switch": "faucet",
    "glass_door": "glass doors",
    "carpet": "carpet",
    "shower_floor": "shower floor",
    "toilet_plunger": "plunger",
    "plug_panel": "power outlet",
    "stand": "stand",
    "potted_plant": "potted plant",
    "poster": "poster",
    "isolation_board": "divider",
    "soap_holder": "soap dish",
    "plug": "power outlet",
    "brush": "hair brush",
    "threshold": "doorframe",
    "air_conditioner_controller": "remote",
    "iron": "iron",
    "ironing_board": "ironing board",
    "safe": "suitcase",
    "gas_cooker": "stove",
    "pressure_cooker": "cooking pot",
    "steamer_pot": "pot",
    "soy_sauce_bottle": "bottle",
    "dishwashing_liquid": "dishwashing soap bottle",
    "water_ladle": "bowl",
    "power_socket_set": "power strip",
    "kitchen_tool_holder": "kitchen cabinet",
    "case": "case",
    "wall_paper": "wall",
    "comb": "hair brush",
    "paper_cutter": "paper cutter",
    "pencil_sharpener": "pen holder",
    "sealing_machine": "machine",
    "poster_board": "poster",
    "shredder": "shredder",
    "footstep": "stair",
    "planter": "plant",
    "floor_light": "lamp",
    "paper_cup": "cup",
    "divider": "divider",
    "hanger": "clothes hanger",
    "glove": "clothing",
    "blanket": "blanket",
    "remote": "remote",
    "cloth": "cloth",
    "clutter": "object",
    "extinguisher": "fire extinguisher",
    "dryer": "clothes dryer",
    "soap_bottle": "soap bottle",
    "fabric_softener_box": "box",
    "dryer_sheet_box": "box",
    "detergent_bottle": "laundry detergent",
    "toaster": "toaster",
    "stacked_bowls": "bowl",
    "pot_lid": "pot",
    "electric_pressure_cooker": "rice cooker",
    "bread": "food display",
    "bagels": "object",
    "oranges": "bananas",
    "card_reader": "card",
    "whiteboard_detergent": "soap dispenser",
    "power_outlet": "power outlet",
    "bouquet": "vase",
    "water_bottle": "water bottle",
    "wall_mounted_telephone": "telephone",
    "fridge": "refrigerator",
    "toy": "toy dinosaur",
    "shoe_box": "box",
    "hole_puncher": "paper cutter",
    "landline_telephone": "telephone",
    "base": "stand",
    "handkerchief": "cloth",
    "cornice_molding": "frame",
    "bathtub_base": "bathtub",
    "bidet": "toilet",
    "pedestal_urinal": "urinal",
    "pedestal_urinal_covered": "urinal",
    "pit_toilet": "toilet",
    "low_wall": "wall",
    "rail": "rail",
    "bottles": "bottles",
    "floor_otherroom": "floor",
    "wall_otherroom": "wall",
    "canopy": "canopy",
    "cable_manager": "cable",
    "sneakers": "shoes",
    "purse": "purse",
    "cushion": "cushion",
    "napkin": "towel",
    "plush_toy": "stuffed animal",
    "adjustable_desk": "desk",
    "tableware": "plates",
    "computer_desk": "desk",
    "cat_kennel": "cat litter box",
    "back_cushion": "pillow",
    "ukulele_bag": "guitar case",
    "litter_box": "trash can",
    "storage_box": "storage bin",
    "toy_doll": "doll",
    "drawer_unit": "drawer",
    "doll": "stuffed animal",
    "laptop_bag": "messenger bag",
    "clothing_rack": "clothing rack",
    "bookshelf": "bookshelves",
    "mask": "cloth",
    "watch": "clock",
    "book": "books",
    "ashtray": "tray",
    "car_key": "car",
    "wallet": "purse",
    "tea_pot": "tea kettle",
    "wire": "cable",
    "rake": "broom",
    "dispenser": "soap dispenser",
    "toilet_tank": "toilet",
    "door_sill": "doorframe",
    "cleanser": "soap",
    "armrest": "armchair",
    "short_wall": "wall",
    "suspended_ceiling": "ceiling",
    "fire_extinguisher_cabinet": "fire extinguisher",
    "plastic_box": "plastic container",
    "sanitation_station": "soap dispenser",
    "plant_pot": "flowerpot",
    "fireplace": "fireplace",
    "computer_table": "desk",
    "tissue_bag": "tissue box",
    "wall_frame": "frame",
    "map_board": "map",
    "automated_teller_machine": "vending machine",
    "ticket": "card",
    "tablet": "ipad",
    "blankets": "blanket",
    "bags": "bag",
    "flag": "flag",
    "blackboard": "blackboard",
    "bar_table": "bar",
    "cardboard_holder": "cardboard",
    "potted_planet": "potted plant",
    "tray": "tray",
    "utensil_holder": "kitchen counter",
    "bird_ceramics": "statue",
    "shirt": "shirt",
    "clothes_rail": "clothes hanger",
    "power_strips": "power strip",
    "card_board": "board",
    "pile_of_blankets": "blanket",
    "bed_net": "bed",
    "umbrella": "umbrella",
    "dragon_fruit": "bananas",
    "tissue": "tissue box",
    "electrical_panel": "electric panel",
    "panel": "door",
    "tube": "tube",
    "pile_of_cloth": "cloth",
    "surface": "table",
    "chair_cushion": "cushion",
    "guide": "book",
    "parapet": "railing",
    "camera": "camera",
    "light_base": "lamp base",
    "first_aid": "object",
    "bench": "bench",
    "potted_plants": "potted plant",
    "pot_cover": "pot",
    "yoga_mat_roll": "yoga mat",
    "panda_doll": "stuffed animal",
    "window_trim": "window",
    "shoe_cabinet": "shoe rack",
    "toilet_paper_holder": "toilet paper dispenser",
    "shower_faucet": "shower faucet handle",
    "bath_sponge": "sponge",
    "ornament": "decoration",
    "planter_box": "plant",
    "cooktop": "stove",
    "knife_block": "knife block",
    "step_stool": "step stool",
    "touchpad": "keyboard",
    "light_box": "light",
    "sound": "speaker",
    "exhaust_fan_vent": "vent",
    "paperbin": "recycling bin",
    "mop_bucket": "bucket",
    "sneaker": "shoes",
    "objects": "object",
    "cd_tray": "cd case",
    "wall_board": "board",
    "room_divider": "divider",
    "paiting": "painting",
    "cabinet_otherroom": "cabinet",
    "electric_switch": "light switch",
    "sign": "exit sign",
    "hand_soap": "soap bottle",
    "window_blinds": "blinds"
}

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

def get_scan_ids(dirname, split):
    filepath = osp.join(dirname, '{}_scans.txt'.format(split))
    scan_ids = np.genfromtxt(filepath, dtype = str)
    return scan_ids

def load_ply_data(data_dir, scan_id):
    """
    Load PLY data and propagate object IDs from faces to vertices.
    """
    filename_in = osp.join(data_dir, scan_id, '{}.ply'.format(scan_id))
    
    if not osp.exists(filename_in):
        raise FileNotFoundError(f"PLY file not found: {filename_in}")
    
    with open(filename_in, 'rb') as file:
        ply_data = PlyData.read(file)
    
    # Extract vertex properties
    x = np.array(ply_data['vertex']['x'])
    y = np.array(ply_data['vertex']['y'])
    z = np.array(ply_data['vertex']['z'])
    red = np.array(ply_data['vertex']['red'])
    green = np.array(ply_data['vertex']['green'])
    blue = np.array(ply_data['vertex']['blue'])
    
    # Extract normals if available
    if 'nx' in ply_data['vertex'] and 'ny' in ply_data['vertex'] and 'nz' in ply_data['vertex']:
        nx = np.array(ply_data['vertex']['nx'])
        ny = np.array(ply_data['vertex']['ny'])
        nz = np.array(ply_data['vertex']['nz'])
        normals = np.stack([nx, ny, nz], axis=-1)
    else:
        normals = None

    # Initialize object IDs for vertices with a default undefined value
    vertex_object_ids = np.full(len(x), -1, dtype='int32')  # Default: -1 (undefined)
    
    # Extract face data
    faces = ply_data['face'].data
    face_vertex_indices = [face['vertex_indices'] for face in faces]
    face_object_ids = [face['objectId'] for face in faces]
    
    # Propagate object IDs to vertices
    for face_indices, obj_id in zip(face_vertex_indices, face_object_ids):
        vertex_object_ids[face_indices] = obj_id  # Assign object ID to all vertices in the face
    
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),       # Coordinates
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),  # Colors
        ('objectId', 'i4')                            # Propagated Object ID
    ]
    
    if normals is not None:
        vertex_dtype.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])  # Normals
    
    vertices = np.empty(len(x), dtype=vertex_dtype)
    
    vertices['x'] = x.astype('f4')
    vertices['y'] = y.astype('f4')
    vertices['z'] = z.astype('f4')
    vertices['red'] = red.astype('u1')
    vertices['green'] = green.astype('u1')
    vertices['blue'] = blue.astype('u1')
    vertices['objectId'] = vertex_object_ids.astype('i4')
    
    if normals is not None:
        vertices['nx'] = normals[:, 0].astype('f4')
        vertices['ny'] = normals[:, 1].astype('f4')
        vertices['nz'] = normals[:, 2].astype('f4')
    
    return vertices

def load_meta_intrinsics(scan_dir, scene_id, stream_type="color_camera"):
    '''
    Load MultiScan intrinsic information
    '''
    meta_intrinsics_path = osp.join(scan_dir, f'{scene_id}.json')
    intrinsics = {}
    
    with open(meta_intrinsics_path,"r") as f:
        json_data=json.load(f)
    
    for stream in json_data.get("streams", []):
        if stream.get("type") == stream_type:
            intrinsic_mat = np.array(stream.get("intrinsics"))
            intrinsic_mat = np.reshape(intrinsic_mat, (3, 3), order='F')
            intrinsics['intrinsic_mat']=intrinsic_mat
            resolution = stream.get("resolution")
            width, height = resolution[1], resolution[0]  # [width, height]
            intrinsics['width']=float(width)
            intrinsics['height']=float(height)
    
    return intrinsics

def load_intrinsics(scan_dir, scene_id, frame_id, stream_type="color_camera"):
    '''
    Load MultiScan intrinsic information
    '''
    intrinsics_path = osp.join(scan_dir, 'poses.jsonl')
    resoultion_path = osp.join(scan_dir, f'{scene_id}.json')
    intrinsics = {}
    
    with open(resoultion_path,"r") as f:
        json_data=json.load(f)
    
    for stream in json_data.get("streams", []):
        if stream.get("type") == stream_type:
            resolution = stream.get("resolution", None)
            if resolution:
                width, height = resolution[1], resolution[0]  # [width, height]
                intrinsics['width']=float(width)
                intrinsics['height']=float(height)
                
        
    with jsonlines.open(intrinsics_path) as reader:
        for entry in reader:
            if entry.get("frame_id") == frame_id:
                intrinsic_mat = np.asarray(entry.get('intrinsics'))
                intrinsic_mat = np.reshape(intrinsic_mat, (3, 3), order='F')
                intrinsics['intrinsic_mat']=intrinsic_mat
                break
    
    return intrinsics

def load_pose(scan_dir, frame_id):
    # Find alignment file
    alignment_path = None
    for file_name in os.listdir(scan_dir):
        if file_name.endswith('.align.json'):
            alignment_path = osp.join(scan_dir, file_name)
            break

    if alignment_path is None:
        raise FileNotFoundError(f"No alignment file found in {scan_dir}")

    with open(alignment_path, "r") as f:
        alignment_data = json.load(f)
    if 'coordinate_transform' not in alignment_data:
        raise ValueError(f"Alignment file {alignment_path} does not contain 'coordinate_transform'")
    coordinate_transform = np.reshape(alignment_data['coordinate_transform'], (4, 4), order='F')
    inv_transform = np.linalg.inv(coordinate_transform)

    pose_path = osp.join(scan_dir, 'poses.jsonl')
    with jsonlines.open(pose_path) as reader:
        for entry in reader:
            if entry.get("frame_id") == frame_id:
                transform = np.asarray(entry.get('transform'))
                transform = np.reshape(transform, (4, 4), order='F')
                transform = np.dot(transform, np.diag([1, -1, -1, 1]))
                transform = transform / transform[3][3]
                aligned_pose = inv_transform @ transform #align camera poses
                return aligned_pose

    raise ValueError(f"Pose for frame_id {frame_id} not found in {pose_path}")


def load_all_poses(scan_dir, frame_idxs):
    frame_poses = {}
    for frame_idx in frame_idxs:
        frame_pose = load_pose(scan_dir, int(frame_idx))
        frame_poses[frame_idx] = frame_pose
    return frame_poses

def load_frame_idxs(scan_dir, skip=None):
    frames_paths = glob(osp.join(scan_dir, 'sequence', '*.jpg'))
    frame_names = [osp.basename(frame_path) for frame_path in frames_paths]
    frame_idxs = [frame_name.split('.')[0].split('-')[-1] for frame_name in frame_names]
    frame_idxs.sort()    

    if skip is None:
        frame_idxs = frame_idxs
    else:
        frame_idxs = [frame_idx for frame_idx in frame_idxs[::skip]]
    return frame_idxs


def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False