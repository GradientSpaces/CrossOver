from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("model")

def build_model(cfg):
    if 'unified' in cfg.model.name.lower():
        model = MODEL_REGISTRY.get(cfg.model.name)(cfg.model, cfg.task.get(cfg.task.name).modalities, cfg.task.get(cfg.task.name).freeze_object_enc)
    else:
        model = MODEL_REGISTRY.get(cfg.model.name)(cfg.model, cfg.task.get(cfg.task.name).modalities)
    return model