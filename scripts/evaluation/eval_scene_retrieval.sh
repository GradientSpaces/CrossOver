export PYTHONWARNINGS="ignore"

# Scene Retrieval Inference
python run_evaluation.py --config-path "$(pwd)/configs/evaluation" --config-name eval_scene.yaml \
task.InferenceSceneRetrieval.val=['Scannet'] \
task.InferenceSceneRetrieval.ckpt_path=/drive/dumps/multimodal-spaces/runs/UnifiedTrain_Scannet+Scan3R/2025-06-19-23:10:12.350698/ckpt/ckpt_200.pth \
hydra.run.dir=. hydra.output_subdir=null 