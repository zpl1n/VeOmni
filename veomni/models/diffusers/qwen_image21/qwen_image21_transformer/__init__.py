"""Register both the VeOmni schema and the official Diffusers config name."""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen_image21_transformer")
@MODEL_CONFIG_REGISTRY.register("QwenImage21Transformer2DModel")
def register_qwen_image21_transformer_config():
    from .configuration_qwen_image21_transformer import QwenImage21Transformer2DConfig

    return QwenImage21Transformer2DConfig


@MODELING_REGISTRY.register("qwen_image21_transformer")
@MODELING_REGISTRY.register("QwenImage21Transformer2DModel")
def register_qwen_image21_transformer_modeling(_architecture: str | None = None):
    from .modeling_qwen_image21_transformer import QwenImage21Transformer2DModel

    return QwenImage21Transformer2DModel
