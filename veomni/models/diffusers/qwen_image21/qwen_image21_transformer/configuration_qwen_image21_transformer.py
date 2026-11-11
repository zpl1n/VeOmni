"""Configuration for the official Qwen-Image 2.1 transformer."""

import diffusers

from veomni.models.configuration_base import VeOmniPretrainedConfig


QWEN_IMAGE21_MODEL_FIELDS = (
    "patch_size",
    "in_channels",
    "out_channels",
    "num_layers",
    "attention_head_dim",
    "num_attention_heads",
    "context_in_dim",
    "mlp_ratio",
    "axes_dims_rope",
    "eps",
    "causal_condition",
)


class QwenImage21Transformer2DConfig(VeOmniPretrainedConfig):
    model_type = "QwenImage21Transformer2DModel"
    condition_model_type = "QwenImage21ConditionModel"

    def __init__(
        self,
        patch_size=1,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        causal_condition=True,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.context_in_dim = context_in_dim
        self.mlp_ratio = mlp_ratio
        self.axes_dims_rope = axes_dims_rope
        self.eps = eps
        self.causal_condition = causal_condition
        super().__init__(**kwargs)
        if patch_size != 1:
            raise ValueError("Qwen-Image 2.1 consumes unpatched latents; patch_size must be 1.")
        if len(axes_dims_rope) != 3 or any(dim <= 0 or dim % 2 for dim in axes_dims_rope):
            raise ValueError("axes_dims_rope must contain three positive even dimensions.")
        if sum(axes_dims_rope) != attention_head_dim:
            raise ValueError("axes_dims_rope must sum to attention_head_dim.")

    def to_diffuser_dict(self):
        return {key: getattr(self, key) for key in QWEN_IMAGE21_MODEL_FIELDS}

    def to_dict(self):
        result = super().to_dict()
        result["_class_name"] = "QwenImage21Transformer2DModel"
        result["_diffusers_version"] = diffusers.__version__
        result.pop("dtype", None)
        return result
