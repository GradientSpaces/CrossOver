export PYTHONWARNINGS="ignore"

# Change val according to the dataset you want to evaluate on

# Instance Baseline
# python run_evaluation.py --config-path "$(pwd)/configs/evaluation" \
# --config-name eval_instance.yaml \
# task.InferenceObjectRetrieval.val=['Scannet'] \
# task.InferenceObjectRetrieval.ckpt_path=/drive/dumps/multimodal-spaces/runs/release_runs/instance_baseline_scannet+scan3r.pth \
# model.name=ObjectLevelEncoder \
# hydra.run.dir=. hydra.output_subdir=null 

# Instance CrossOver
python run_evaluation.py --config-path "$(pwd)/configs/evaluation" \
--config-name eval_instance.yaml \
task.InferenceObjectRetrieval.val=['ARKitScenes'] \
task.InferenceObjectRetrieval.ckpt_path=/drive/dumps/multimodal-spaces/runs/new_runs/instance_crossover_scannet+scan3r+multiscan+arkitscenes.pth \
model.name=SceneLevelEncoder \
hydra.run.dir=. hydra.output_subdir=null 