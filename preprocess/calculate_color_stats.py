import numpy as np
import os
import os.path as osp
import yaml
from tqdm import tqdm
import argparse
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

def load_config(config_path):
    """Load and resolve the YAML config file."""
    config = OmegaConf.load(config_path)
    # Resolve variable substitutions
    config = OmegaConf.to_container(config, resolve=True)
    return config

def get_train_scan_ids(dataset_name, base_dir):
    """Get train scan IDs for each dataset based on their file structure."""
    train_scan_ids = []
    
    if dataset_name.lower() == 'scannet':
        train_file = osp.join(base_dir, 'files', 'scannetv2_train.txt')
        if osp.exists(train_file):
            with open(train_file, 'r') as f:
                train_scan_ids = [line.strip() for line in f.readlines()]
        else:
            print(f"Warning: Train split file not found for ScanNet: {train_file}")
    
    elif dataset_name.lower() == 'scan3r':
        train_file = osp.join(base_dir, 'files', 'train_scans.txt')
        if osp.exists(train_file):
            with open(train_file, 'r') as f:
                train_scan_ids = [line.strip() for line in f.readlines()]
        else:
            print(f"Warning: Train split file not found for Scan3R: {train_file}")
    
    elif dataset_name.lower() == 'multiscan':
        train_file = osp.join(base_dir, 'files', 'train_scans.txt')
        if osp.exists(train_file):
            with open(train_file, 'r') as f:
                train_scan_ids = [line.strip() for line in f.readlines()]
        else:
            print(f"Warning: Train split file not found for MultiScan: {train_file}")
    
    elif dataset_name.lower() == 'arkitscenes':
        train_file = osp.join(base_dir, 'files', 'train_scans.txt')
        if osp.exists(train_file):
            with open(train_file, 'r') as f:
                train_scan_ids = [line.strip() for line in f.readlines()]
        else:
            print(f"Warning: Train split file not found for ARKitScenes: {train_file}")

    elif dataset_name.lower() == 'structured3d':
        train_file = osp.join(base_dir, 'files', 'train_scans.txt')
        if osp.exists(train_file):
            with open(train_file, 'r') as f:
                train_scan_ids = [line.strip() for line in f.readlines()]
        else:
            print(f"Warning: Train split file not found for Structured3D: {train_file}")
            
    return train_scan_ids

def compute_color_stats_for_dataset(dataset_name, dataset_config, train_scan_ids=None):
    """
    Compute color statistics for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'Scannet', 'Scan3R')
        dataset_config: Dataset configuration from YAML
        train_scan_ids: List of train scan IDs (optional)
    """
    
    process_dir = dataset_config['process_dir']
    base_dir = dataset_config['base_dir']
    
    print(f"\n{'='*60}")
    print(f"PROCESSING DATASET: {dataset_name}")
    print(f"{'='*60}")
    print(f"Base directory: {base_dir}")
    print(f"Process directory: {process_dir}")
    
    # Get train scan IDs if not provided
    if train_scan_ids is None:
        train_scan_ids = get_train_scan_ids(dataset_name, base_dir)
    
    print(f"Total train scans to process: {len(train_scan_ids)}")
    
    # Filter to only existing processed scans
    valid_train_scans = []
    for scan_id in train_scan_ids:
        data_path = osp.join(process_dir, 'scans', scan_id, 'data3D.npz')
        if osp.exists(data_path):
            valid_train_scans.append(scan_id)

    if len(valid_train_scans) == 0:
        print(f"No valid processed train scans found for {dataset_name}")
        return None
    
    # Collect color statistics
    all_color_means = []
    all_color_second_moments = []
    
    
    for scan_id in tqdm(valid_train_scans, desc=f"Processing {dataset_name}"):
        try:
            data_path = osp.join(process_dir, 'scans', scan_id, 'data3D.npz')
            data = np.load(data_path, allow_pickle=True)
            
            # Handle different data structures
            scene_data = data['scene'].item()
            mesh_colors = scene_data['pcl_feats']
            
            # Normalize colors to [0, 1] range
            colors_normalized = mesh_colors / 255.0
            
            # Compute per-scan statistics
            color_mean = colors_normalized.mean(axis=0)  # E[X]
            color_second_moment = (colors_normalized ** 2).mean(axis=0)  # E[X²]
            
            all_color_means.append(color_mean)
            all_color_second_moments.append(color_second_moment)
            
        except Exception as e:
            print(f"Error processing {scan_id}: {e}")
            continue
    
    if len(all_color_means) == 0:
        print(f"No valid color data found for {dataset_name}")
        return None
    
    # Compute global statistics
    global_color_mean = np.array(all_color_means).mean(axis=0)
    global_color_second_moment = np.array(all_color_second_moments).mean(axis=0)
    global_color_std = np.sqrt(global_color_second_moment - global_color_mean**2)
    
    # Prepare output
    color_mean_std = {
        'mean': [float(val) for val in global_color_mean],
        'std': [float(val) for val in global_color_std]
    }
    
    # Save to dataset's process directory
    output_path = osp.join(process_dir, 'color_mean_std.yaml')
    os.makedirs(process_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(color_mean_std, f, default_flow_style=False, indent=2)
    
    print(f"\nStatistics computed from {len(all_color_means)} train scans:")
    print(f"Mean (RGB): [{global_color_mean[0]:.6f}, {global_color_mean[1]:.6f}, {global_color_mean[2]:.6f}]")
    print(f"Std  (RGB): [{global_color_std[0]:.6f}, {global_color_std[1]:.6f}, {global_color_std[2]:.6f}]")
    print(f"Saved to: {output_path}")
    
    return color_mean_std

def main():
    parser = argparse.ArgumentParser(description="Compute color statistics for all datasets from process_3d.yaml config")
    parser.add_argument("--config-path", type=str, 
                       default="/Users/gauravpradeep/CrossOver_ScaleUp/configs/preprocess",
                       help="Path to config directory")
    parser.add_argument("--config-name", type=str, 
                       default="process_3d.yaml",
                       help="Config file name")
    parser.add_argument("--datasets", type=str, nargs="*",
                       help="Specific datasets to process (e.g., Scannet Scan3R). If not specified, processes all.")
    
    args = parser.parse_args()
    
    # Load configuration
    config_file = osp.join(args.config_path, args.config_name)
    if not osp.exists(config_file):
        print(f"Config file not found: {config_file}")
        return
    
    config = load_config(config_file)
    
    data_config = config['data']
    available_datasets = []

    for key, value in data_config.items():
        if key.lower() == 'front3d':
            continue
        if isinstance(value, dict) and 'process_dir' in value and 'base_dir' in value:
            available_datasets.append(key)
        
    if args.datasets:
        datasets_to_process = [d for d in args.datasets if d in available_datasets]
        if not datasets_to_process:
            print(f"None of the specified datasets found in config: {args.datasets}")
            return
    else:
        datasets_to_process = available_datasets
    
    print(f"Processing datasets: {datasets_to_process}")
    
    # Process each dataset
    results = {}
    for dataset_name in datasets_to_process:
        dataset_config = data_config[dataset_name]
        
        try:
            result = compute_color_stats_for_dataset(dataset_name, dataset_config)
            results[dataset_name] = result
        except Exception as e:
            print(f"Failed to process {dataset_name}: {e}")
            results[dataset_name] = None
    
    # Summary
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, result in results.items():
        if result:
            print(f"✓ {dataset_name}: Successfully computed color statistics")
        else:
            print(f"✗ {dataset_name}: Failed to compute color statistics")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()