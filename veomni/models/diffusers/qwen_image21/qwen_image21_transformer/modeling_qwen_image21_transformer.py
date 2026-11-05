"""Native Qwen-Image 2.1 DiT, following the DiffSynth-Studio single-stream model."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from numbers import Integral

import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME as DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME as DIFFUSERS_SAFE_WEIGHTS_NAME
from torch import nn
from transformers import initialization
from transformers.modeling_outputs import ModelOutput
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.modeling_base import VeOmniPreTrainedModel
from veomni.ops import VeomniOp

from .configuration_qwen_image21_transformer import QWEN_IMAGE21_MODEL_FIELDS, QwenImage21Transformer2DConfig
from .layers import (
    QwenImage21OutputNorm,
    QwenImage21RMSNorm,
    QwenImage21Rope,
    QwenImage21SwiGLU,
    QwenImage21TextProjection,
    QwenImage21TimeProjection,
    apply_rotary,
    select_modulation,
)
from .parallel import joint_attention, joint_forward


_SUPPORTED_ATTENTION = frozenset({"eager", "sdpa", "veomni_sdpa"})


def eager_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Small-tensor reference for the model-local attention handle."""
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        else:
            scores = scores + attention_mask.float()
    fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
    scores = scores.masked_fill(fully_masked, 0.0)
    weights = torch.softmax(scores, dim=-1).masked_fill(fully_masked, 0.0).to(value.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), None


class QwenImage21Attention(nn.Module):
    is_causal = False
    num_key_value_groups = 1

    def __init__(self, config, dim):
        super().__init__()
        self.config = config
        self.heads = config.num_attention_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = QwenImage21RMSNorm(config.attention_head_dim, config.eps)
        self.norm_k = QwenImage21RMSNorm(config.attention_head_dim, config.eps)
        self.attention = VeomniOp("attention", "standard", config.ops_config.attn_implementation)

    @staticmethod
    def _attend(attn, query, key, value, mask):
        if mask is not None and query.device.type == "npu" and mask.shape[-2] == 1:
            # Ascend SDPA requires the query dimension to be explicit in a dense mask.
            mask = mask.expand(*mask.shape[:-2], query.shape[1], -1).contiguous()
        return attn.attention(
            attn,
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            mask,
            dropout=0.0,
            is_causal=False,
            skip_ulysses=True,
        )[0]

    def forward(self, hidden_states, rotary, segments, key_valid):
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        query = apply_rotary(self.norm_q(query).to(value.dtype), rotary)
        key = apply_rotary(self.norm_k(key).to(value.dtype), rotary)
        output = joint_attention(self._attend, self, query, key, value, segments, key_valid)
        return self.to_out[1](self.to_out[0](output.flatten(2).to(query.dtype)))


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, eps=config.eps, elementwise_affine=False)
        self.attn = QwenImage21Attention(config, dim)
        self.img_norm2 = nn.LayerNorm(dim, eps=config.eps, elementwise_affine=False)
        self.img_mlp = QwenImage21SwiGLU(dim, dim * config.mlp_ratio)

    def forward(self, hidden_states, modulation, rotary, target_mask, segments, key_valid):
        scale_attn, gate_attn, scale_mlp, gate_mlp = (
            select_modulation(chunk, target_mask) for chunk in modulation.chunk(4, dim=-1)
        )
        attention_input = self.img_norm1(hidden_states) * (1 + scale_attn)
        hidden_states = hidden_states + gate_attn.tanh() * self.attn(attention_input, rotary, segments, key_valid)
        mlp_input = self.img_norm2(hidden_states) * (1 + scale_mlp)
        hidden_states = hidden_states + gate_mlp.tanh() * self.img_mlp(mlp_input)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp(-65504, 65504)
        return hidden_states


@dataclass
class QwenImage21ModelOutput(ModelOutput):
    loss: dict[str, torch.Tensor] | None = None
    predictions: list[torch.Tensor] | None = None


class QwenImage21Transformer2DModel(VeOmniPreTrainedModel):
    config_class = QwenImage21Transformer2DConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["QwenImage21TransformerBlock"]

    def __init__(self, config, **kwargs):
        attention = kwargs.pop("attn_implementation", None)
        if attention is not None:
            config._attn_implementation = attention
        kwargs.pop("torch_dtype", None)
        kwargs.pop("trust_remote_code", None)
        super().__init__(config, **kwargs)
        self.config.tie_word_embeddings = False
        if config.num_attention_heads % get_parallel_state().ulysses_size:
            raise ValueError("Qwen-Image 2.1 attention heads must be divisible by ulysses_size.")

        dim = config.num_attention_heads * config.attention_head_dim
        self.pos_embed = QwenImage21Rope(config.axes_dims_rope)
        self.time_text_embed = QwenImage21TimeProjection(dim)
        self.txt_in = QwenImage21TextProjection(config.context_in_dim, dim, config.eps)
        self.img_in = nn.Linear(config.in_channels, dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            QwenImage21TransformerBlock(config, dim) for _ in range(config.num_layers)
        )
        self.norm_out = QwenImage21OutputNorm(dim, config.eps)
        self.proj_out = nn.Linear(dim, config.out_channels or config.in_channels, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, QwenImage21RMSNorm):
            if module.zero_centered:
                initialization.zeros_(module.weight)
            else:
                initialization.ones_(module.weight)
        else:
            super()._init_weights(module)

    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False, **kwargs):
        if attn_implementation not in _SUPPORTED_ATTENTION:
            raise ValueError(
                f"Unsupported Qwen-Image 2.1 attention {attn_implementation!r}; choose eager, sdpa, or veomni_sdpa."
            )
        return attn_implementation

    @staticmethod
    def _normalize_img_shapes(img_shapes):
        if isinstance(img_shapes, torch.Tensor):
            img_shapes = img_shapes.cpu().tolist()
        if not isinstance(img_shapes, (list, tuple)) or not img_shapes:
            raise ValueError("img_shapes must contain condition-image grids followed by the target grid.")
        if len(img_shapes) == 3 and all(isinstance(value, Integral) for value in img_shapes):
            img_shapes = [[img_shapes]]
        elif len(img_shapes[0]) == 3 and all(isinstance(value, Integral) for value in img_shapes[0]):
            img_shapes = [img_shapes]
        result = [[tuple(int(value) for value in shape) for shape in sample] for sample in img_shapes]
        if any(len(shape) != 3 or shape[0] != 1 or min(shape) <= 0 for sample in result for shape in sample):
            raise ValueError("Qwen-Image 2.1 img_shapes must be positive (1, height, width) image grids.")
        return result

    def predict_noise(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        img_shapes,
        img_mask,
        encoder_hidden_states_mask=None,
        return_dict=True,
    ):
        img_shapes = self._normalize_img_shapes(img_shapes)
        target_tokens = prod(img_shapes[0][-1])
        if target_tokens % 4:
            raise ValueError("The target grid must contain a multiple of four latent tokens.")
        if img_mask.ndim != 2 or img_mask.shape[1] != encoder_hidden_states.shape[1] + target_tokens // 4:
            raise ValueError("img_mask must include the prompt positions followed by target_tokens / 4 image slots.")
        if not bool(img_mask[:, -target_tokens // 4 :].all()):
            raise ValueError("The trailing img_mask slots must all represent the target image.")
        if len(img_shapes) != hidden_states.shape[0] or any(shape != img_shapes[0] for shape in img_shapes):
            raise ValueError("A tensor batch must have the same image layout; use list batches for different layouts.")
        if not torch.equal(img_mask.bool(), img_mask[:1].bool().expand_as(img_mask)):
            raise ValueError("A tensor batch must share img_mask; use list batches for different layouts.")
        if hidden_states.shape[1] != sum(prod(shape) for shape in img_shapes[0]):
            raise ValueError("hidden_states must contain all condition-image latents followed by target latents.")
        output = joint_forward(
            self,
            hidden_states.to(self.dtype),
            encoder_hidden_states.to(self.dtype),
            timestep,
            img_shapes,
            img_mask.bool(),
            encoder_hidden_states_mask,
        )
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        img_shapes,
        img_mask,
        training_target=None,
        loss_weight=None,
        encoder_hidden_states_mask=None,
        latents=None,
        return_dict=True,
    ):
        if training_target is None:
            return self.predict_noise(
                hidden_states,
                timestep,
                encoder_hidden_states,
                img_shapes,
                img_mask,
                encoder_hidden_states_mask,
                return_dict=return_dict,
            )
        fields = (hidden_states, timestep, encoder_hidden_states, img_shapes, img_mask, training_target)
        if isinstance(hidden_states, list):
            count = len(hidden_states)
            if not count or any(not isinstance(field, list) or len(field) != count for field in fields):
                raise ValueError("All Qwen-Image 2.1 training fields must have the same nonzero sample count.")
            masks = [None] * count if encoder_hidden_states_mask is None else encoder_hidden_states_mask
            if not isinstance(masks, list) or len(masks) != count:
                raise ValueError("encoder_hidden_states_mask must match the training sample count.")
        else:
            fields = tuple([field] for field in fields)
            masks = [encoder_hidden_states_mask]
        count = len(fields[0])
        weights = [None] * count if loss_weight is None else loss_weight
        if not isinstance(weights, list):
            weights = [weights]
        if len(weights) != count:
            raise ValueError("loss_weight must match the training sample count.")
        predictions, losses = [], []
        for sample, step, embeddings, shapes, slots, target, mask, weight in zip(*fields, masks, weights):
            prediction = self.predict_noise(sample, step, embeddings, shapes, slots, mask, return_dict=False)[0]
            if prediction.shape != target.shape:
                raise ValueError("Qwen-Image 2.1 prediction and training_target shapes must match exactly.")
            predictions.append(prediction)
            loss = F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)
            if weight is not None:
                loss = loss * weight.to(device=loss.device, dtype=loss.dtype)
            losses.append(loss)
        return QwenImage21ModelOutput(loss={"mse_loss": torch.cat(losses).mean()}, predictions=predictions)

    def save_pretrained(self, path, **kwargs):
        # Reuse the Diffusers serializer for standalone export. A meta shell
        # shares the live tensors and preserves the published file layout.
        from diffusers import QwenImage21Transformer2DModel as DiffusersTransformer

        with torch.device("meta"):
            export = DiffusersTransformer(**self.config.to_diffuser_dict())
        export.register_to_config(ops_implementation=dict(self.config.ops_implementation))
        export.load_state_dict(self.state_dict(), strict=True, assign=True)
        return export.save_pretrained(path, **kwargs)

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        hub_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "revision", "token", "local_files_only", "subfolder")
            if key in kwargs
        }
        config = kwargs.pop("config", None)
        overrides = {key: kwargs.pop(key) for key in QWEN_IMAGE21_MODEL_FIELDS if key in kwargs}
        ops = kwargs.pop("ops_implementation", None)
        attention = kwargs.pop("attn_implementation", None)
        config_kwargs = {"ops_implementation": ops, "attn_implementation": attention, **overrides}
        if config is None:
            config = cls.config_class.from_pretrained(path, **hub_kwargs, **config_kwargs)
        elif isinstance(config, Mapping):
            config = cls.config_class.from_dict(dict(config), **config_kwargs)
        elif isinstance(config, cls.config_class):
            config = copy.deepcopy(config)
            config.update(overrides)
            config._apply_ops_overrides(ops_implementation=ops, attn_implementation=attention)
        else:
            raise TypeError("config must be a QwenImage21Transformer2DConfig or a mapping.")

        if path is None or "state_dict" in kwargs:
            return super().from_pretrained(path, *args, config=config, **kwargs)

        variant = kwargs.get("variant")
        lookup_kwargs = {"_raise_exceptions_for_missing_entries": False, **hub_kwargs}
        if kwargs.get("use_safetensors") is not False:
            for filename in (SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME):
                if variant is not None:
                    stem, extension = filename.rsplit(".", 1)
                    filename = f"{stem}.{variant}.{extension}"
                if cached_file(path, filename, **lookup_kwargs) is not None:
                    return super().from_pretrained(path, *args, config=config, **kwargs)
            if variant is None:
                for filename in (DIFFUSERS_SAFE_WEIGHTS_NAME, DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME):
                    if cached_file(path, filename, **lookup_kwargs) is not None:
                        config.transformers_weights = filename
                        return super().from_pretrained(path, *args, config=config, **kwargs)

        # Legacy Diffusers variants and binary weights use Diffusers' resolver.
        # Standard safetensors above stream directly into the native model.
        from diffusers import QwenImage21Transformer2DModel as DiffusersTransformer

        dispatch = {
            key: kwargs.pop(key)
            for key in ("device_map", "max_memory", "offload_folder", "offload_buffers")
            if key in kwargs
        }
        loaded = DiffusersTransformer.from_pretrained(path, **config.to_diffuser_dict(), **kwargs)
        source = loaded[0] if isinstance(loaded, tuple) else loaded
        model = super().from_pretrained(
            None, *args, config=config, state_dict=source.state_dict(), dtype=source.dtype, **dispatch
        )
        return (model, loaded[1]) if isinstance(loaded, tuple) else model
