# Dataset Preparation

## Overview

This document provides instructions for pre-processing different datasets, including 
- ScanNet
- 3RScan
- ARKitScenes
- MultiScan

## Prerequisites

### Environment
Before you begin, simply activate the `crossover` conda environment.

### Download the Data

#### Original Data
- **ScanNet**: Download ScanNet v2 data from the [official website](https://github.com/ScanNet/ScanNet), we use the official training and validation split from [here](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark).

- **3RScan**: Download 3RScan dataset from the [official website](https://github.com/WaldJohannaU/3RScan), we use the official (full list of scan ids including reference + rescans) training split from [here](https://campar.in.tum.de/public_datasets/3RScan/train_scans.txt) and validation split from [here](https://campar.in.tum.de/public_datasets/3RScan/val_scans.txt).
    - Download `3RScan.json` from [here](https://campar.in.tum.de/public_datasets/3RScan/3RScan.json) and `objects.json` from [here](https://campar.in.tum.de/public_datasets/3DSSG/3DSSG/objects.json).
    - Download the class mapping file `3RScan.v2 Semantic Classes - Mapping.csv` from [here](https://docs.google.com/spreadsheets/d/1eRTJ2M9OHz7ypXfYD-KTR1AIT-CrVLmhJf8mxgVZWnI/edit?gid=0#gid=0).

- **ShapeNet**: Download ShapenetCore dataset from the [official Huggingface release](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) and unzip.

- **MultiScan**: Download MultiScan dataset from the [official website](https://github.com/smartscenes/multiscan).

- **ARKitScenes**: Download ARKitScenes dataset from the [official website](https://github.com/apple/ARKitScenes).


#### Referral and CAD annotations
We use [SceneVerse](https://scene-verse.github.io/) for instance referrals (ScanNet, 3RScan, MultiScan, & ARKitScenes) and [Scan2CAD](https://github.com/skanti/Scan2CAD) for CAD annotations (ScanNet). 

- **SceneVerse** - Download the Scannet and 3RScan data under `annotations/refer` from the [official website](https://scene-verse.github.io/).
- **Scan2CAD** - Download `full_annotations.json` from the [official website](https://github.com/skanti/Scan2CAD?tab=readme-ov-file#download-dataset).

### Prepare The Data
Exact instructions for data setup + preparation below:

#### ScanNet
1. Run the following to extract ScanNet data 
```bash
cd scannet
python preprocess_2d_scannet.py --scannet_path PATH_TO_SCANNET --output_path PATH_TO_SCANNET
python unzip_scannet.py --scannet_path PATH_TO_SCANNET --output_path PATH_TO_SCANNET
```

2. To have a unified structure of objects `objects.json` like provided in `3RScan`, run the following:

```bash
cd scannet
python scannet_objectdata.py
```

> Change `base_dataset_dir` to `Scannet` dataset root directory.

2. Move the relevant files from `Sceneverse` and `Scannet` under `files/`. Once completed, the data structure would look like the following:

```
Scannet/
├── scans/
│   ├── scene0000_00/
│   │   ├── data/
│   │   │    ├── color/
│   │   |    ├── depth/
|   |   |    ├── instance-filt/
│   │   |    └── pose/
|   |   ├── intrinsics.txt
│   │   ├── scene0000_00_vh_clean_2.ply 
|   |   ├── scene0000_00_vh_clean_2.labels.ply
|   |   ├── scene0000_00_vh_clean_2.0.010000.segs.json
|   |   ├── scene0000_00_vh_clean.aggregation.json
|   |   └── scene0000_00_2d-instance-filt.zip
|   └── ...
└── files
    ├── scannetv2_val.txt
    ├── scannetv2_train.txt
    ├── scannetv2-labels.combined.tsv
    ├── scan2cad_full_annotations.json
    ├── objects.json
    └── sceneverse  
        └── ssg_ref_rel2_template.json
```

#### 3RScan

1. Run the following to align the re-scans and reference scans in the same coordinate system & unzip `sequence.zip` for every scan:

```bash
cd scan3r
python align_scan.py  (change `root_scan3r_dir` to `PATH_TO_SCAN3R`)
python unzip_scan3r.py --scan3r_path PATH_TO_SCAN3R --output_path PATH_TO_SCAN3R
```

2. Move the relevant files from `Sceneverse` and `3RScan` under `files/`.

Once completed, the data structure would look like the following:

```
Scan3R/
├── scans/
│   ├── 20c993b5-698f-29c5-85a5-12b8deae78fb/
│   │   ├── sequence/ (folder containing frame-wise color + depth + pose information)
|   |   ├── labels.instances.align.annotated.v2.ply
│   │   └── labels.instances.annotated.v2.ply
|   └── ...
└── files
    ├── 3RScan.json
    ├── 3RScan.v2 Semantic Classes - Mapping.csv
    ├── objects.json
    ├── train_scans.txt
    ├── val_scans.txt
    └── sceneverse  
        └── ssg_ref_rel2_template.json
```

#### ARKitScenes
1. Download ARKitScenes 3dod data using the following command:

```bash
python download_data.py 3dod --video_id_csv PATH_TO_3dod_train_val_splits.csv --download_dir PATH_TO_ARKITSCENES
```
The files mentioned in the above command - ```download_data.py``` and ```3dod_train_val_splits.csv``` can be found in the official repository [here](https://github.com/apple/ARKitScenes), along with more detailed instructions and descriptions of the data.

2. Once the data is downloaded, run the following to organize it as per our requirements.
 
 ```bash
cd ARKitScenes
mv 3dod/Training/* scans
mv 3dod/Validation/* scans
```

3. Move the relevant files from `Sceneverse` and `ARKitScenes` under `files/`.

Once completed, the data structure would look like the following:
```
ARKitScenes/
├── scans/
│   ├── 40753679/
│   │   ├── 40753679_frames/ 
│   │   │    ├── lowres_depth/ (folder containing depth images)
│   │   │    ├── lowres_wide/ (folder containing rgb images)
│   │   │    ├── lowres_wide_intrinsics/ (folder containing frame wise camera intrinsics)
│   │   │    ├── lowres_wide.traj (camera trajectory)
│   │   ├── 40753679_3dod_annotation.json
│   │   ├── 40753679_3dod_mesh.ply
|   └── 
└── files
    ├── scannetv2-labels.combined.tsv
    ├── train_scans.txt
    ├── val_scans.txt
    ├── metadata.csv
    ├── 3dod_train_val_splits.csv
    └── sceneverse  
        └── ssg_ref_rel2_template.json
```

#### MultiScan
1. Download MultiScan data into MultiScan/scenes and run the following to extract MultiScan data.
 
 ```bash
cd MultiScan/scenes
unzip '*.zip'
rm -rf '*.zip'
```
3. To generate sequence of RGB images and corresponding camera poses from the ```.mp4``` file, run the following:
```bash
cd prepare_data/multiscan
python preprocess_2d_multiscan.py --base_dir PATH_TO_MULTISCAN --frame_interval {frame_interval}
```
Once completed, the data structure would look like the following:
```
MultiScan/
├── scenes/
│   ├── scene_00000_00/
│   │   ├── sequence/ (folder containing rgb images at specified frame interval)
|   |   ├── frame_ids.txt
│   │   ├── scene_00000_00.annotations.json
│   │   ├── scene_00000_00.jsonl
│   │   ├── scene_00000_00.confidence.zlib
│   │   ├── scene_00000_00.mp4
│   │   ├── poses.jsonl
│   │   ├── scene_00000_00.ply
│   │   ├── scene_00000_00.align.json
│   │   ├── scene_00000_00.json
|   └── 
└── files
    ├── scannetv2-labels.combined.tsv
    ├── train_scans.txt
    ├── test_scans.txt
    └── sceneverse  
        └── ssg_ref_rel2_template.json
```