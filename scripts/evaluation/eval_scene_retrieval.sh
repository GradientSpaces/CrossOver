export PYTHONWARNINGS="ignore"

# Scene Retrieval Inference
python run_evaluation.py --config-path "$(pwd)/configs/evaluation" --config-name eval_scene.yaml \
task.InferenceSceneRetrieval.val=['Scan3R'] \
task.InferenceSceneRetrieval.ckpt_path=/drive/dumps/multimodal-spaces/runs/new_runs/scene_crossover_scannet+scan3r_norm+scratch.pth \
hydra.run.dir=. hydra.output_subdir=null 