export PYTHONWARNINGS="ignore"

# Scene Retrieval Inference
python run_evaluation.py --config-path "$(pwd)/configs/evaluation" --config-name eval_scene.yaml \
task.InferenceSceneRetrieval.val=['ARKitScenes'] \
task.InferenceSceneRetrieval.ckpt_path=/drive/dumps/multimodal-spaces/runs/curr_runs/scene_crossover_scannet+scan3r+multiscan+arkitscenes.pth \
hydra.run.dir=. hydra.output_subdir=null 