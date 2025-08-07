from fvcore.common.registry import Registry

PROCESSOR_REGISTRY = Registry("Processor")

def build_processor(processor_name, data_config, modality_config, split):
    print(f"Building processor: {processor_name}")
    processor = PROCESSOR_REGISTRY.get(processor_name)(data_config, modality_config, split)
    return processor