from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("model")

def build_model(cfg):
    model = MODEL_REGISTRY.get(cfg.model.name)(cfg.model, cfg.task.get(cfg.task.name).modalities)
    return model