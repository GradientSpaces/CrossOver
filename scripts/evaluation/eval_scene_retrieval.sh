export PYTHONWARNINGS="ignore"

# Scene Retrieval Inference
python run_evaluation.py --config-path "$(pwd)/configs/evaluation" --config-name eval_scene.yaml \
task.InferenceSceneRetrieval.val=['ARKitScenes'] \
task.InferenceSceneRetrieval.ckpt_path=/drive/dumps/multimodal-spaces/runs/UnifiedTrain_Scannet+Scan3R+MultiScan+ARKitScenes/2025-07-03-07:39:02.553100/ckpt/best.pth \
hydra.run.dir=. hydra.output_subdir=null 