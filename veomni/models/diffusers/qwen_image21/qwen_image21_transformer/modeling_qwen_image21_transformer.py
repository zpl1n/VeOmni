"""Qwen-Image 2.1 training with native checkpoint keys and block-causal attention.

The backbone and its zero-centered norms are the pinned Diffusers implementation.
Only attention dispatch, NPU rotary embeddings, and the training loss are adapted.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from numbers import Integral
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from diffusers import QwenImage21Transformer2DModel as _QwenImage21Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope
from transformers.modeling_outputs import ModelOutput

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.modeling_base import VeOmniPreTrainedModel as PreTrainedModel
from veomni.ops import VeomniOp

from .configuration_qwen_image21_transformer import QWEN_IMAGE21_INIT_SIGNATURE, QwenImage21Transformer2DConfig


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


class QwenImage21AttentionKernelModule:
    """Metadata view expected by the framework attention operators."""

    def __init__(self, attn, implementation):
        self.config = SimpleNamespace(_attn_implementation=implementation)
        self.is_causal = False
        self.num_key_value_groups = 1
        self.training = attn.training


class QwenImage21DeviceRope(QwenImage21Rope):
    """Compute Ascend's positional tables on CPU, transferring only real values."""

    def forward(self, img_shapes, image_pad_mask, device):
        if device.type != "npu":
            return super().forward(img_shapes, image_pad_mask, device)
        frequencies = super().forward(img_shapes, image_pad_mask.cpu(), torch.device("cpu"))
        return torch.view_as_real(frequencies).contiguous().to(device)


def apply_qwen_image21_rotary_emb(hidden_states, frequencies):
    """Use real interleaved rotation on every accelerator, including Ascend."""
    if frequencies.is_complex():
        frequencies = torch.view_as_real(frequencies)
    real, imaginary = hidden_states.float().unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = frequencies.float().unbind(-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return (
        torch.stack((real * cos - imaginary * sin, real * sin + imaginary * cos), dim=-1)
        .flatten(-2)
        .to(hidden_states.dtype)
    )


class QwenImage21AttnProcessor:
    """Exact segment attention: causal text, bidirectional individual image blocks."""

    def __init__(self, config):
        self.implementation = config.ops_config.attn_implementation
        if self.implementation not in _SUPPORTED_ATTENTION:
            raise ValueError(
                f"Unsupported Qwen-Image 2.1 attention {self.implementation!r}; choose eager, sdpa, or veomni_sdpa."
            )
        self.veomni_attn = VeomniOp("attention", "standard", self.implementation)

    def _attention(self, attn, query, key, value, mask):
        if mask is not None and query.device.type == "npu" and mask.shape[-2] == 1:
            # Ascend's fused SDPA needs an explicit query dimension in a dense mask.
            mask = mask.expand(*mask.shape[:-2], query.shape[1], -1).contiguous()
        return self.veomni_attn(
            QwenImage21AttentionKernelModule(attn, self.implementation),
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            mask,
            dropout=0.0,
            is_causal=False,
            skip_ulysses=True,
        )[0]

    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        rotary_emb=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ):
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query, key = attn.norm_q(query).to(value.dtype), attn.norm_k(key).to(value.dtype)
        if rotary_emb is not None:
            query = apply_qwen_image21_rotary_emb(query, rotary_emb)
            key = apply_qwen_image21_rotary_emb(key, rotary_emb)
        if layer_cache is not None:
            if kv_cache_mode == "extract" and cache_write_slice is not None:
                layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
            elif kv_cache_mode == "cached":
                cached_key, cached_value = layer_cache.get()
                key = torch.cat((cached_key, key), dim=1)
                value = torch.cat((cached_value, value), dim=1)

        if segments is None:
            output = self._attention(attn, query, key, value, attention_mask)
        else:
            outputs = []
            for start, end, is_text in segments:
                mask = None
                if is_text:
                    # A text query sees earlier segments and the causal part of its own segment.
                    query_positions = torch.arange(start, end, device=query.device)
                    key_positions = torch.arange(end, device=query.device)
                    mask = (key_positions[None, :] <= query_positions[:, None])[None, None]
                if key_valid is not None:
                    valid = key_valid[:, None, None, :end]
                    mask = valid if mask is None else (mask & valid)
                outputs.append(self._attention(attn, query[:, start:end], key[:, :end], value[:, :end], mask))
            prefix_length = segments[-1][1] if segments else 0
            target_mask = None if key_valid is None else key_valid[:, None, None, :]
            outputs.append(self._attention(attn, query[:, prefix_length:], key, value, target_mask))
            output = torch.cat(outputs, dim=1)
        output = output.flatten(2, 3).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](output))


@dataclass
class QwenImage21ModelOutput(ModelOutput):
    loss: dict[str, torch.Tensor] | None = None
    predictions: list[torch.Tensor] | None = None


class _QwenImage21InitShim(_QwenImage21Transformer2DModel):
    def __init__(self, *args, **kwargs):
        # HF's cooperative init must not build the full default 7B backbone.
        torch.nn.Module.__init__(self)


class QwenImage21Transformer2DModel(PreTrainedModel, _QwenImage21InitShim):
    config_class = QwenImage21Transformer2DConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["QwenImage21TransformerBlock"]

    def __init__(self, config, **kwargs):
        implementation = kwargs.pop("attn_implementation", None)
        if implementation is not None:
            config._attn_implementation = implementation
        kwargs.pop("torch_dtype", None)
        kwargs.pop("trust_remote_code", None)
        PreTrainedModel.__init__(self, config, **kwargs)
        del self._internal_dict
        construction_device = torch.get_default_device()
        # Upstream creates complex positional tables while building the backbone.
        # CPU construction avoids unsupported complex factories inside an NPU device context.
        with torch.device("cpu"):
            _QwenImage21Transformer2DModel.__init__(self, **config.to_diffuser_dict())
        self.config = config
        self.config.tie_word_embeddings = False
        self._install_veomni_attention()
        if construction_device.type != "cpu" and not any(parameter.is_meta for parameter in self.parameters()):
            self.to(construction_device)

    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False, **kwargs):
        if attn_implementation not in _SUPPORTED_ATTENTION:
            raise ValueError(
                f"Unsupported Qwen-Image 2.1 attention {attn_implementation!r}; choose eager, sdpa, or veomni_sdpa."
            )
        return attn_implementation

    @property
    def config(self):
        return self._internal_dict

    @config.setter
    def config(self, value):
        self._internal_dict = value

    def _install_veomni_attention(self):
        if get_parallel_state().sp_enabled:
            raise ValueError("Qwen-Image 2.1 sequence parallelism is not implemented; set Ulysses and CP sizes to 1.")
        # Positional tables are non-parameter data; keep their initialization on CPU even under meta init.
        with torch.device("cpu"):
            self.pos_embed = QwenImage21DeviceRope(theta=10000, axes_dim=list(self.config.axes_dims_rope))
        for block in self.transformer_blocks:
            block.attn.set_processor(QwenImage21AttnProcessor(self.config))

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
        if get_parallel_state().sp_enabled:
            raise ValueError("Qwen-Image 2.1 sequence parallelism is not implemented.")
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
        output = _QwenImage21Transformer2DModel.forward(
            self,
            hidden_states=hidden_states.to(self.dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.dtype),
            timestep=timestep,
            img_shapes=img_shapes,
            img_mask=img_mask.bool(),
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            return_dict=False,
        )[0][:, -target_tokens:]
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
        original_config = self.config
        self.config = {**self.config.to_diffuser_dict(), "ops_implementation": dict(self.config.ops_implementation)}
        try:
            _QwenImage21Transformer2DModel.save_pretrained(self, path, **kwargs)
        finally:
            self.config = original_config

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        ops = kwargs.pop("ops_implementation", None)
        attention = kwargs.pop("attn_implementation", None)
        config = kwargs.pop("config", None)
        overrides = {key: kwargs[key] for key in QWEN_IMAGE21_INIT_SIGNATURE.parameters if key in kwargs}
        if isinstance(config, cls.config_class):
            config = copy.deepcopy(config)
            for key, value in overrides.items():
                setattr(config, key, value)
            config._apply_ops_overrides(ops_implementation=ops, attn_implementation=attention)
        elif config is None or isinstance(config, Mapping):
            config_kwargs = {"ops_implementation": ops, "attn_implementation": attention, **overrides}
            if config is None:
                hub_kwargs = {
                    key: kwargs[key]
                    for key in ("cache_dir", "revision", "token", "local_files_only", "subfolder")
                    if key in kwargs
                }
                config = cls.config_class.from_pretrained(path, **hub_kwargs, **config_kwargs)
            else:
                config = cls.config_class.from_dict(dict(config), **config_kwargs)
        else:
            raise TypeError("config must be a QwenImage21Transformer2DConfig or a mapping.")
        if config.ops_config.attn_implementation not in _SUPPORTED_ATTENTION:
            raise ValueError(
                f"Unsupported Qwen-Image 2.1 attention {config.ops_config.attn_implementation!r}; choose eager, sdpa, or veomni_sdpa."
            )
        loaded = _QwenImage21Transformer2DModel.from_pretrained(path, **{**config.to_diffuser_dict(), **kwargs})
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        model.__class__ = cls
        model.config = config
        model.config.tie_word_embeddings = False
        model._install_veomni_attention()
        return loaded
