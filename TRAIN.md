# :weight_lifting: Training and Inference

#### Environment Setup
Follow setup instructions from README. 
```bash
$ conda activate crossover
```

#### Train Instance Baseline
Adjust path parameters in `configs/train/train_instance_baseline.yaml` and run the following:

```bash
$ bash scripts/train/train_instance_baseline.sh
```

#### Train Instance Retrieval Pipeline
Adjust path parameters in `configs/train/train_instance_crossover.yaml` and run the following:

```bash
$ bash scripts/train/train_instance_crossover.sh
```

#### Train Scene Retrieval Pipeline
Adjust path/configuration parameters in `configs/train/train_scene_crossover.yaml`. You can also add your customised dataset or choose to train on any combination of Scannet, 3RScan, ARKitScenes & MultiScan. Run the following:

```bash
$ bash scripts/train/train_scene_crossover.sh
```

> The scene retrieval pipeline uses the trained weights from instance retrieval pipeline (for object feature calculation), please ensure to update `task:UnifiedTrain:object_enc_ckpt` in the config file when training.

#### Checkpoint Inventory
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
|Instance CrossOver trained on ScanNet + 3RScan + ARKitScenes + MultiScan        | [ScanNet+3RScan+ARKitScenes+MultiScan]() |

##### ```scene_crossover```
| Description            | Checkpoint Link |
| ------------------ | -------------- |
| Unified CrossOver trained on ScanNet + 3RScan        | [ScanNet+3RScan](https://huggingface.co/gradient-spaces/CrossOver/tree/main/scene_crossover_scannet%2Bscan3r.pth) |
| Unified CrossOver trained on ScanNet + 3RScan + ARKitScenes + MultiScan       | [ScanNet+3RScan+ARKitScenes+MultiScan]() |


# :shield: Single Inference
We release script to perform inference (generate scene-level embeddings) on a single scan of 3RScan/Scannet. Detailed usage in the file. Quick instructions below:

```bash
$ python single_inference/scene_inference.py
```

Various configurable parameters:

- `--dataset`: dataset name, Scannet/Scan3R
- `--data_dir`: data directory (eg: `./datasets/Scannet`, assumes similar structure as in `preprocess.md`).
- `--floorplan_dir`: directory consisting of the rasterized floorplans (this can point to the downloaded preprocessed directory), only for Scannet
- `--ckpt`: Path to the pre-trained scene crossover model checkpoint (details [here](TRAIN.md#checkpoint-inventory)), example_path: `./checkpoints/scene_crossover_scannet+scan3r.pth/`).
- `--scan_id`: the scan id from the dataset you'd like to calculate embeddings for (if not provided, embeddings for all scans are calculated).

The script will output embeddings in the same format as provided [here](DATA.md/#generated-embedding-data).

# :bar_chart: Evaluation
#### Cross-Modal Object Retrieval
Run the following script (refer to the script to run instance baseline/instance crossover) for object instance + scene retrieval results using the instance-based methods. Detailed usage inside the script.

```bash
$ bash scripts/evaluation/eval_object_retrieval.sh
```

> Running this script for 3RScan dataset will also show point-to-point temporal instance matching results on the RIO category subset.

#### Cross-Modal Scene Retrieval
Run the following script (for scene crossover). Detailed usage inside the script.

```bash
$ bash scripts/evaluation/eval_scene_retrieval.sh
```
