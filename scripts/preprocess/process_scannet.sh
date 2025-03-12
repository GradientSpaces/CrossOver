export PYTHONWARNINGS="ignore"

# Preprocessing Object Level + Scene Level + Unified Data
python preprocessor.py --config-path "$(pwd)/configs/preprocess" --config-name process_3d.yaml data.sources=['Scannet'] hydra.run.dir=. hydra.output_subdir=null
python preprocessor.py --config-path "$(pwd)/configs/preprocess" --config-name process_2d.yaml data.sources=['Scannet'] hydra.run.dir=. hydra.output_subdir=null 
python preprocessor.py --config-path "$(pwd)/configs/preprocess" --config-name process_1d.yaml data.sources=['Scannet']  hydra.run.dir=. hydra.output_subdir=null
# Multi-modal dumping
python preprocessor.py --config-path "$(pwd)/configs/preprocess" --config-name process_multimodal.yaml data.sources=['Scannet'] hydra.run.dir=. hydra.output_subdir=null