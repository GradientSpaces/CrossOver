export PYTHONWARNINGS="ignore"

# Preprocessing Object Level + Scene Level + Unified Data
python3 preprocessor.py --config-path /Users/gauravpradeep/CrossOver_ScaleUp/configs/preprocess --config-name process_3d.yaml data.sources=['ARKitScenes'] hydra.run.dir=. hydra.output_subdir=null 
python3 preprocessor.py --config-path /Users/gauravpradeep/CrossOver_ScaleUp/configs/preprocess --config-name process_1d.yaml data.sources=['ARKitScenes'] hydra.run.dir=. hydra.output_subdir=null 
python3 preprocessor.py --config-path /Users/gauravpradeep/CrossOver_ScaleUp/configs/preprocess --config-name process_2d.yaml data.sources=['ARKitScenes'] hydra.run.dir=. hydra.output_subdir=null 

# # Multi-modal dumping
python3 preprocessor.py --config-path /Users/gauravpradeep/CrossOver_ScaleUp/configs/preprocess --config-name process_multimodal.yaml data.sources=['ARKitScenes'] hydra.run.dir=. hydra.output_subdir=null 
