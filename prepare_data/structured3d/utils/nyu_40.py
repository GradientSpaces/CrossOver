"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
from typing import Dict


class NYU40:
    """
    NYU40 label definition and color scheme
    """
    LABEL_DICT: Dict[str, int] = dict()
    INDEX_DICT: Dict[int, str] = dict()

    @staticmethod
    def load_dict(i2l: bool):
        """
        Load global label dictionary
        """
        if not __class__.LABEL_DICT:
            label_path = os.path.join(os.path.dirname(os.path.abspath(\
                __file__)), 'label_mapping.txt')
            with open(label_path, encoding='utf-8') as l_fp:
                for line in l_fp.readlines():
                    items = line.rstrip('\n').split('\t')
                    __class__.LABEL_DICT[items[-1]] = int(items[0])
                    __class__.INDEX_DICT[int(items[0])] = items[-1]
        return __class__.INDEX_DICT if i2l else __class__.LABEL_DICT

    @staticmethod
    def label_to_index(label: str):
        """
        Mapping index to label
        """
        return __class__.load_dict(False)[label]

    @staticmethod
    def index_to_label(index: int):
        """
        Mapping index to label
        """
        return __class__.load_dict(True)[index]

    @staticmethod
    def color_scheme():
        """
        Get the color coding scheme
        Source from: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util.py
        Copyright: ScanNet
        """
        return [
            (0, 0, 0),
            (174, 199, 232),		# wall
            (152, 223, 138),		# floor
            (31, 119, 180), 		# cabinet
            (255, 187, 120),		# bed
            (188, 189, 34), 		# chair
            (140, 86, 75),  		# sofa
            (255, 152, 150),		# table
            (214, 39, 40),  		# door
            (197, 176, 213),		# window
            (148, 103, 189),		# bookshelf
            (196, 156, 148),		# picture
            (23, 190, 207), 		# counter
            (178, 76, 76),
            (247, 182, 210),		# desk
            (66, 188, 102),
            (219, 219, 141),		# curtain
            (140, 57, 197),
            (202, 185, 52),
            (51, 176, 203),
            (200, 54, 131),
            (92, 193, 61),
            (78, 71, 183),
            (172, 114, 82),
            (255, 127, 14), 		# refrigerator
            (91, 163, 138),
            (153, 98, 156),
            (140, 153, 101),
            (158, 218, 229),		# shower curtain
            (100, 125, 154),
            (178, 127, 135),
            (120, 185, 128),
            (146, 111, 194),
            (44, 160, 44),  		# toilet
            (112, 128, 144),		# sink
            (96, 207, 209),
            (227, 119, 194),		# bathtub
            (213, 92, 176),
            (94, 106, 211),
            (82, 84, 163),  		# other furn
            (100, 85, 144)
        ]