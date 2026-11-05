"""Register the local Qwen-Image 2.1 condition config and model."""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("QwenImage21ConditionModel")
def register_qwen_image21_condition_config():
    from .configuration_qwen_image21_condition import QwenImage21ConditionModelConfig

    return QwenImage21ConditionModelConfig


@MODELING_REGISTRY.register("QwenImage21ConditionModel")
def register_qwen_image21_condition_modeling(_architecture: str | None = None):
    from .modeling_qwen_image21_condition import QwenImage21ConditionModel

    return QwenImage21ConditionModel
