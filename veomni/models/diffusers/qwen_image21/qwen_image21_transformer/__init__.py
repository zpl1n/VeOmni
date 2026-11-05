"""Register the local Qwen-Image 2.1 transformer config and model."""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("QwenImage21Transformer2DModel")
def register_qwen_image21_transformer_config():
    from .configuration_qwen_image21_transformer import QwenImage21Transformer2DConfig

    return QwenImage21Transformer2DConfig


@MODELING_REGISTRY.register("QwenImage21Transformer2DModel")
def register_qwen_image21_transformer_modeling(_architecture: str | None = None):
    from .modeling_qwen_image21_transformer import QwenImage21Transformer2DModel

    return QwenImage21Transformer2DModel
