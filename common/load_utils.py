import json
import os
import os.path as osp
import pickle
import re
from pathlib import Path
from typing import Any
import torch 
import random
import numpy as np
import yaml

def make_dir(dir_path: str) -> None:
    """Creates a directory if it does not exist."""
    if not Path(dir_path).exists():
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists; creates it if it does not.
    """
    if not osp.exists(path):
        os.makedirs(path)

def assert_dir(path: str) -> None:
    """Asserts that a directory exists."""
    assert osp.exists(path)

def load_pkl_data(filename: str) -> Any:
    """Loads data from a pickle file."""
    with open(filename, 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict

def write_pkl_data(data_dict: Any, filename: str) -> None:
    """Writes data to a pickle file."""
    with open(filename, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_json(filename: str) -> Any:
    """Loads data from a JSON file."""
    file = open(filename)
    data = json.load(file)
    file.close()
    return data

def load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.safe_load(f)
        return file

def write_json(data_dict: Any, filename: str) -> None:
    """Writes data to a JSON file with indentation."""
    json_obj = json.dumps(data_dict, indent=4)
 
    with open(filename, "w") as outfile:
        outfile.write(json_obj)

def load_npz_as_dict(filename: str) -> dict:
    with np.load(filename, allow_pickle=True) as npz:
        if isinstance(npz, np.lib.npyio.NpzFile):
            out = {}
            for k in npz.files:                
                val = npz[k]                 
                if (isinstance(val, np.ndarray) and
                    val.dtype == object and
                    val.shape == ()):
                    out[k] = val.item()        
                else:
                    out[k] = val               
            return out                   

def get_print_format(value: Any) -> str:
    """Determines the appropriate format string for a given value."""
    if isinstance(value, int):
        return 'd'
    if isinstance(value, str):
        return 's'
    if value == 0:
        return '.3f'
    if value < 1e-6:
        return '.3e'
    if value < 1e-3:
        return '.6f'
    return '.6f'


def get_format_strings(kv_pairs: list) -> list:
    """Generates format strings for a list of key-value pairs."""
    log_strings = []
    for key, value in kv_pairs:
        fmt = get_print_format(value)
        format_string = '{}: {:' + fmt + '}'
        log_strings.append(format_string.format(key, value))
    return log_strings

def get_first_index_batch(x: Any) -> Any:
    """Retrieves the first index from a batch, handling different data types."""
    if isinstance(x, list):
        x = x[0]
    elif isinstance(x, torch.Tensor):
        x = x.squeeze(0)
    elif isinstance(x, dict):
        x = {key: get_first_index_batch(value) for key, value in x.items()}
    return x

def split_sentence(sentence: str) -> list:
    """Splits a sentence into individual sentences based on periods."""
    sentence = re.split(r'[.]', sentence)
    sentence = [s.strip() for s in sentence]
    sentence = [s for s in sentence if len(s) > 0]
    return sentence

def set_random_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False