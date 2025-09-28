<p align="center">
  <h2 align="center"> CrossOver: 3D Scene Cross-Modal Alignment </h2>
  <p align="center">
    <a href="https://sayands.github.io/">Sayan Deb Sarkar</a><sup>1</sup>
    .
    <a href="https://miksik.co.uk/">Ondrej Miksik</a><sup>2</sup>
    .
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>2, 3</sup>
    .
    <a href="https://www.linkedin.com/in/d%C3%A1niel-bar%C3%A1th-3a489092/">Dániel Béla Baráth</a><sup>3, 4</sup>
    .
    <a href="https://ir0.github.io/">Iro Armeni</a><sup>1</sup>
  </p>
  <p align="center"> <strong>Computer Vision And Pattern Recognition (CVPR) 2025</strong></p>
  <p align="center">
    <sup>1</sup>Stanford University · <sup>2</sup>Microsoft Spatial AI Lab · <sup>3</sup>ETH Zürich · <sup>4</sup>HUN-REN SZTAKI
  </p>
  <h3 align="center">

 [![arXiv](https://img.shields.io/badge/arXiv-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2502.15011) 
 [![ProjectPage](https://img.shields.io/badge/Project_Page-CrossOver-blue)](https://sayands.github.io/crossover)
 [![Hugging Face (LCM) Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/gradient-spaces/CrossOver)
 [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
 <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="https://github.com/sayands/crossover/blob/main/static/videos/teaser.gif" width="100%">
  </a>
</p>

## 📃 Abstract

Multi-modal 3D object understanding has gained significant attention, yet current approaches often rely on rigid object-level modality alignment or 
assume complete data availability across all modalities. We present **CrossOver**, a novel framework for cross-modal 3D scene understanding via flexible, scene-level modality alignment. Unlike traditional methods that require paired data for every object instance, CrossOver learns a unified, modality-agnostic embedding space for scenes by aligning modalities - RGB images, point clouds, CAD models, floorplans, and text descriptions - without explicit object semantics. Leveraging dimensionality-specific encoders, a multi-stage training pipeline, and emergent cross-modal behaviors, CrossOver supports robust scene retrieval and object localization, even with missing modalities. Evaluations on ScanNet and 3RScan datasets show its superior performance across diverse metrics, highlighting CrossOver’s adaptability for real-world applications in 3D scene understanding.

### 🚀 Features

- Flexible Scene-Level Alignment 🌐 - Aligns RGB, point clouds, CAD, floorplans, and text at the scene level— no perfect data needed!
- Emergent Cross-Modal Behaviors 🤯 - Learns unseen modality pairs (e.g., floorplan ↔ text) without explicit pairwise training.
- Real-World Applications 🌍 AR/VR, robotics, construction—handles temporal changes (e.g., object rearrangement) effortlessly.

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
  <li>
      <a href="#hammer_and_wrench-installation">Installation</a>
    </li>
    <li>
      <a href="#arrow_down-data">Data</a>
    </li>
    <li>
      <a href="#film_projector-demo">Demo</a>
    </li>
    <li>
      <a href="#weight_lifting-training-and-inference">Training & Inference</a>
    </li>
    <li>
      <a href="#pray-acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#page_facing_up-citation">Citation</a>
    </li>
  </ol>
</details>


# :newspaper: News
- ![](https://img.shields.io/badge/New!-8A2BE2) **Version 1.0** - **CrossOver is now stronger than ever**. We recommend updating to this version; changes include:
  - More powerful pre-trained checkpoints; now available on Huggingface 👉 [here](https://huggingface.co/gradient-spaces/CrossOver/tree/main).
  - Support for 2 additional datasets - ARKitScenes & MultiScan
  

- [2025-05] Pretrained checkpoints have been moved to HuggingFace 👉 [here](https://huggingface.co/gradient-spaces/CrossOver/tree/main).
- [2025-03] CrossOver is accepted to **CVPR 2025** as **Highlight**. 🔥
- [2025-02] **Version 0.1** - We release CrossOver on arXiv with codebase + pre-trained checkpoints. Checkout our [paper](https://arxiv.org/abs/2502.15011) and [website](https://sayands.github.io/crossover/).

# :hammer_and_wrench: Installation
The code has been tested on: 
```yaml
Ubuntu: 22.04 LTS
Python: 3.9.20
CUDA: 12.1
GPU: GeForce RTX 4090/RTX 3090
```

## 📦 Setup

Clone the repo and setup as follows:

```bash
$ git clone git@github.com:GradientSpaces/CrossOver.git
$ cd CrossOver
$ conda env create -f req.yml
$ conda activate crossover
```

Further installation for `MinkowskiEngine`, `Pointnet2_PyTorch` and `GPU kNN` (for I2P-MAE setup). Setup as follows:

```bash
$ git clone --recursive "https://github.com/EthenJ/MinkowskiEngine"
$ conda install openblas-devel -c anaconda

# Minkowski Engine
$ cd MinkowskiEngine/ && python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --force_cuda --blas=openblas

# Pointnet2_PyTorch
$ cd .. && git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
$ pip install pointnet2_ops_lib/.

# GPU kNN
$ cd .. && pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

> Since we use CUDA 12.1, we use the above `MinkowskiEngine` fork; for other CUDA drivers, please refer to the official [repo](https://github.com/NVIDIA/MinkowskiEngine).

# :arrow_down: Data
See [DATA.MD](DATA.md) for detailed instructions on data download, preparation and preprocessing. We list the data used in the current version of CrossOver in the table below:


| Dataset Name | Object Modality               | Scene Modality                      | Object Temporal Information | Scene Temporal Information
| ------------ | ----------------------------- | ----------------------------------- |  -------------------------- | -------------------------- |
| Scannet      | `[point, rgb, cad, referral]` | `[point, rgb, floorplan, referral]` |    ❌                       |          ✅                |
| 3RScan       | `[point, rgb, referral]`      | `[point, rgb, referral]`            |    ✅                       |          ✅                |
| ARKitScenes       | `[point, rgb, referral]`      | `[point, rgb, referral]`            |    ❌                       |          ✅                |
| MultiScan       | `[point, rgb, referral]`      | `[point, rgb, referral]`            |    ❌                       |          ✅                |

> To run our scene retrieval demo, you only need to download generated embedding data; no need for any data preprocessing. Running the instance retrieval demo for 3RScan, MultiScan and ARKitScenes requires generating 'gt-projection-seg.npz', containing framewise 2D instance segmentation, as described in our dataset preprocessing instructions.

# :film_projector: Demo

## Instance Retrieval Demo

This demo script allows users to process a custom object and run cross-modal retrieval to find the closest matched object within a target scene . Detailed usage can be found inside the script. Example usage below:

```bash
$ python demo/demo_instance_retrieval.py
```

Various configurable parameters:

- `--query_path`: Path to query object(point cloud, image, or text) 
- `--query_modality`: Query modality - Options: `point`, `rgb`, `referral` 
- `--scan_id`: Scene ID to search in 
- `--target_modality`: Target modality to match against - Options: `point`, `rgb`, `referral`, `cad` 
- `--dataset`: Dataset name - Options: `scannet`, `scan3r`, `arkitscenes`, `multiscan` 
- `--data_dir`: Path to dataset directory - default: `/drive/datasets/Scannet`
- `--process_dir`: Path to preprocessed features directory (for gt-projection-seg.npz)
- `--ckpt`: Path to model checkpoint 
- `--top_k`: Number of top results to return - default: `5`


## Scene Retrieval Demo

This demo script allows users to process a custom scene and retrieve the closest match from the supported datasets using different modalities. Detailed usage can be found inside the script. Example usage below:

```bash
$ python demo/demo_scene_retrieval.py
```

Various configurable parameters:

- `--query_path`: Path to the query scene file (eg: `./example_data/dining_room/scene_cropped.ply`).
- `--database_path`: Path to the precomputed embeddings of the database scenes downloaded before (eg: `./release_data/embed_scannet.pt`).
- `--query_modality`: Modality of the query scene, Options: `point`, `rgb`, `floorplan`, `referral`
- `--database_modality`: Modality used for retrieval. Same options as above.
- `--ckpt`: Path to the pre-trained scene crossover model checkpoint (details [here](#checkpoints)), example_path: `./checkpoints/scene_crossover_scannet+scan3r.pth/`).

For embedding and pre-trained model download, refer to [generated embedding data](DATA.md#generated-embedding-data) and [checkpoints](#checkpoints) sections.

> [!TIP]
> We also provide scripts for inference on a single scan of the supported datasets. Details in **Single Inference** section in [TRAIN.md](TRAIN.md).


# :weight_lifting: Training and Inference 

See [TRAIN.md](TRAIN.md) for the inventory of available checkpoints and detailed instructions on training and inference/evaluation with pre-trained checkpoints. The checkpoint inventory is listed below:

#### Checkpoints
We provide all available checkpoints on huggingface 👉 [here](https://huggingface.co/gradient-spaces/CrossOver/tree/main). Detailed descriptions in the table below:

##### ```instance_baseline```
| Description            | Checkpoint Link |
| ------------------ | -------------- |
|Instance Baseline trained on 3RScan        | [3RScan](https://huggingface.co/gradient-spaces/CrossOver/tree/main/instance_baseline_scan3r.pth) |
|Instance Baseline trained on ScanNet        | [ScanNet](https://huggingface.co/gradient-spaces/CrossOver/tree/main/instance_baseline_scannet.pth) |
|Instance Baseline trained on ScanNet + 3RScan        | [ScanNet+3RScan](https://huggingface.co/gradient-spaces/CrossOver/tree/main/instance_baseline_scannet%2Bscan3r.pth) |

##### ```instance_crossover```
| Description            | Checkpoint Link |
| ------------------ | -------------- |
|Instance CrossOver trained on 3RScan        | [3RScan](https://huggingface.co/gradient-spaces/CrossOver/tree/main/instance_crossover_scan3r.pth) |
|Instance CrossOver trained on ScanNet        | [ScanNet](https://huggingface.co/gradient-spaces/CrossOver/tree/main/instance_crossover_scannet.pth) |
|Instance CrossOver trained on ScanNet + 3RScan        | [ScanNet+3RScan](https://huggingface.co/gradient-spaces/CrossOver/tree/main/instance_crossover_scannet%2Bscan3r.pth) |
|Instance CrossOver trained on ScanNet + 3RScan + ARKitScenes + MultiScan        | [ScanNet+3RScan+ARKitScenes+MultiScan](https://huggingface.co/gradient-spaces/CrossOver/tree/main/instance_crossover_scannet%2Bscan3r%2Bmultiscan%2Barkitscenes.pth) |

##### ```scene_crossover```
| Description            | Checkpoint Link |
| ------------------ | -------------- |
| Unified CrossOver trained on ScanNet + 3RScan        | [ScanNet+3RScan](https://huggingface.co/gradient-spaces/CrossOver/tree/main/scene_crossover_scannet%2Bscan3r.pth) |
| Unified CrossOver trained on ScanNet + 3RScan + ARKitScenes + MultiScan        | [ScanNet+3RScan+ARKitScenes+MultiScan](https://huggingface.co/gradient-spaces/CrossOver/tree/main/scene_crossover_scannet%2Bscan3r%2Bmultiscan%2Barkitscenes.pth) |


## 🚧 TODO List
- [x] Release evaluation on temporal instance matching
- [x] Release inference on single scan cross-modal object retrieval

## 📧 Contact
If you have any questions regarding this project, please use the github issue tracker or contact Sayan Deb Sarkar (sdsarkar@stanford.edu).

# :pray: Acknowledgements
We thank the authors from [3D-VisTa](https://github.com/3d-vista/3D-VisTA), [SceneVerse](https://github.com/scene-verse/sceneverse) and [SceneGraphLoc](https://github.com/y9miao/VLSG) for open-sourcing their codebases.

# :page_facing_up: Citation

```bibtex
@inproceedings{sarkar2025crossover,
author={Sayan Deb Sarkar and Ondrej Miksik and Marc Pollefeys and Daniel Barath and Iro Armeni},
title={CrossOver: 3D Scene Cross-Modal Alignment}, 
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2025}
}
```
