"""Diffusers 0.37 checkpoint I/O for the native DiT."""

from diffusers import ConfigMixin, ModelMixin

from .configuration_qwen_image21_transformer import QWEN_IMAGE21_MODEL_FIELDS, QwenImage21Transformer2DConfig


class QwenImage21Transformer2DModel(ModelMixin, ConfigMixin):
    """Expose the native module names to the Diffusers serializer and resolver."""

    def __init__(self, model):
        super().__init__()
        for name, module in model.named_children():
            self.add_module(name, module)
        self.register_to_config(
            **model.config.to_diffuser_dict(), ops_implementation=dict(model.config.ops_implementation)
        )

    @classmethod
    def from_config(cls, config, **overrides):
        from .modeling_qwen_image21_transformer import QwenImage21Transformer2DModel as NativeTransformer

        fields = {
            k: v
            for k, v in {**config, **overrides}.items()
            if k in QWEN_IMAGE21_MODEL_FIELDS or k == "ops_implementation"
        }
        return cls(NativeTransformer(QwenImage21Transformer2DConfig(**fields)))
