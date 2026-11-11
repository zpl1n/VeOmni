# Copyright 2026 Qwen-Image Team, The HuggingFace Team, and Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native Qwen-Image 2.1 DiT, following the DiffSynth-Studio single-stream model.

Parameter names match the released Diffusers checkpoint. Rotary frequencies use
real pairs for Ascend.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain, groupby, repeat
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
from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    slice_input_tensor,
)
from veomni.models.modeling_base import VeOmniPreTrainedModel
from veomni.ops import VeomniOp

from .configuration_qwen_image21_transformer import QWEN_IMAGE21_MODEL_FIELDS, QwenImage21Transformer2DConfig


_SUPPORTED_ATTENTION = frozenset({"eager", "sdpa", "veomni_sdpa"})
_LATENT_TOKENS_PER_IMAGE_SLOT = 4


class QwenImage21Rope(nn.Module):
    """Three-axis RoPE with centered image coordinates and sequential text."""

    def __init__(self, axes_dims):
        super().__init__()
        # Keep deterministic tables outside the state dict and meta/FSDP buffers.
        positions = torch.cat([torch.arange(8192, device="cpu"), torch.arange(-1024, 0, device="cpu")])
        self.frequencies = []
        for dim in axes_dims:
            inverse = 1.0 / torch.pow(10000, torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
            angles = torch.outer(positions, inverse)
            self.frequencies.append(torch.view_as_real(torch.polar(torch.ones_like(angles), angles)))

    def forward(self, img_shapes, image_tokens, device):
        frame, height, width = [], [], []
        cursor = position = 0
        for _, h, w in img_shapes:
            start = image_tokens.index(True, cursor)
            text = list(range(position, position + start - cursor))
            position += len(text)
            frame.extend(text)
            height.extend(text)
            width.extend(text)
            frame.extend([position] * (h * w))
            height.extend(row for row in range(h // 2 - h, h // 2) for _ in range(w))
            width.extend(list(range(w // 2 - w, w // 2)) * h)
            position += max(h, w)
            cursor = start + h * w
        text = list(range(position, position + len(image_tokens) - cursor))
        frame.extend(text)
        height.extend(text)
        width.extend(text)
        table_device = torch.device("cpu") if device.type == "npu" else device
        self.frequencies = [freq.to(table_device) for freq in self.frequencies]
        tables = [
            freq[torch.tensor(ids, device=table_device)] for freq, ids in zip(self.frequencies, (frame, height, width))
        ]
        return torch.cat(tables, dim=-2).to(device)


def apply_rotary(hidden_states, frequencies):
    pairs = hidden_states.float().unflatten(-1, (-1, 2))
    if hidden_states.device.type in ("cpu", "cuda"):
        rotated = torch.view_as_complex(pairs) * torch.view_as_complex(frequencies).unsqueeze(1)
        return torch.view_as_real(rotated).flatten(-2).to(hidden_states.dtype)
    real, imaginary = pairs.unbind(-1)
    cos, sin = frequencies[None, :, None].unbind(-1)
    rotated = torch.stack((real * cos - imaginary * sin, real * sin + imaginary * cos), dim=-1)
    return rotated.flatten(-2).to(hidden_states.dtype)


class QwenImage21TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(256, dim, bias=False)
        self.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, features):
        return self.linear_2(F.silu(self.linear_1(features)))


class QwenImage21TimeProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = QwenImage21TimestepEmbedding(dim)
        self.frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32, device="cpu") / 128)

    def forward(self, timestep, hidden_states):
        angles = (1000.0 * timestep.float())[:, None] * self.frequencies.to(timestep.device)
        features = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.timestep_embedder(features.to(hidden_states.dtype))


class QwenImage21RMSNorm(nn.Module):
    def __init__(self, dim, eps, *, zero_centered=False):
        super().__init__()
        self.eps = eps
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(torch.zeros(dim) if zero_centered else torch.ones(dim))

    def forward(self, hidden_states):
        values = hidden_states.float()
        inverse_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        if self.zero_centered:
            return (values * inverse_rms * (1 + self.weight.float())).to(hidden_states.dtype)
        normalized = hidden_states * inverse_rms
        return normalized.to(self.weight.dtype) * self.weight


class QwenImage21TextProjection(nn.Module):
    def __init__(self, input_dim, dim, eps):
        super().__init__()
        self.text_norm = QwenImage21RMSNorm(input_dim, eps, zero_centered=True)
        self.in_layer = nn.Linear(input_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, context):
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(context)), approximate="tanh"))


class QwenImage21SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)
        self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, hidden_states):
        return self.out(F.silu(self.gate_layer(hidden_states)) * self.proj(hidden_states))


def select_modulation(params, target_mask):
    if target_mask is None:
        return params.unsqueeze(1)
    return torch.where(target_mask.view(1, -1, 1), params[:-1, None], params[-1:, None])


class QwenImage21OutputNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, hidden_states, temb, target_mask):
        scale = self.linear(F.silu(temb).to(hidden_states.dtype))
        return self.norm(hidden_states) * (1 + select_modulation(scale, target_mask))


def _prefix_segments(image_tokens, image_lengths):
    """Split the prefix into causal text runs and separate bidirectional images."""
    if sum(image_lengths) != sum(image_tokens):
        raise ValueError("img_shapes must account for all image tokens in img_mask.")
    image_ids = chain.from_iterable(repeat(index, length) for index, length in enumerate(image_lengths[:-1]))
    prefix_ids = (next(image_ids) if is_image else -1 for is_image in image_tokens[: -image_lengths[-1]])
    segments = []
    start = 0
    for image_id, tokens in groupby(prefix_ids):
        end = start + sum(1 for _ in tokens)
        segments.append((start, end, image_id < 0))
        start = end
    return segments


def joint_forward(model, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask, text_mask):
    """Interleave text and latents before running the Transformer blocks."""
    hidden_states = model.img_in(hidden_states)
    context = model.txt_in(encoder_hidden_states)
    repeats = torch.where(img_mask[0], _LATENT_TOKENS_PER_IMAGE_SLOT, 1)
    image_mask = torch.repeat_interleave(img_mask[0], repeats)
    image_tokens = image_mask.tolist()
    image_lengths = [prod(shape) for shape in img_shapes[0]]
    target_tokens = image_lengths[-1]
    segments = _prefix_segments(image_tokens, image_lengths)
    target_slots = target_tokens // _LATENT_TOKENS_PER_IMAGE_SLOT
    joint = torch.cat([context, context.new_zeros(context.shape[0], target_slots, context.shape[2])], dim=1)
    joint = joint.repeat_interleave(repeats, dim=1)
    joint[:, image_mask] = hidden_states
    length = joint.shape[1]
    prefix_length = length - target_tokens
    rotary = model.pos_embed(img_shapes[0], image_tokens, device=hidden_states.device)
    timestep = timestep.to(hidden_states.dtype)
    modulation_mask = None
    if model.config.causal_condition:
        timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        modulation_mask = torch.arange(length, device=joint.device) >= prefix_length
    temb = model.time_text_embed(timestep, hidden_states)
    modulation = model.modulation(temb)
    key_valid = None
    if text_mask is not None:
        key_valid = torch.ones(context.shape[0], length, dtype=torch.bool, device=joint.device)
        text_positions = (~image_mask).nonzero(as_tuple=True)[0]
        vlm_text = ~img_mask[0][: text_mask.shape[1]]
        key_valid[:, text_positions] = text_mask.bool()[:, vlm_text]
    parallel = get_parallel_state()
    if parallel.ulysses_enabled:
        joint = slice_input_tensor(joint, dim=1, group=parallel.ulysses_group)
        rotary = slice_input_tensor(rotary, dim=0, group=parallel.ulysses_group)
        if modulation_mask is not None:
            modulation_mask = slice_input_tensor(modulation_mask, dim=0, group=parallel.ulysses_group)
    for block in model.transformer_blocks:
        args = (joint, modulation, rotary, modulation_mask, segments, key_valid, length)
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            joint = model._gradient_checkpointing_func(block, *args)
        else:
            joint = block(*args)
    output = model.proj_out(model.norm_out(joint, temb, modulation_mask))
    if parallel.ulysses_enabled:
        # Every SP rank computes the full loss. Gather backward sums its
        # replicated gradients before FSDP averages over DP and SP ranks.
        output = gather_outputs(output, gather_dim=1, group=parallel.ulysses_group)
    return output[:, prefix_length:length]


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
        attention = config.ops_config.attn_implementation
        # DiffSynth lets PyTorch select the SDPA kernel for each block mask.
        self.attention = VeomniOp("attention", "standard", "sdpa" if attention == "veomni_sdpa" else attention)

    def _attend(self, query, key, value, mask):
        if mask is not None and query.device.type == "npu" and mask.shape[-2] == 1:
            # Ascend SDPA requires the query dimension to be explicit in a dense mask.
            mask = mask.expand(*mask.shape[:-2], query.shape[1], -1).contiguous()
        return self.attention(
            self,
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            mask,
            dropout=0.0,
            is_causal=False,
            skip_ulysses=True,
        )[0]

    def forward(self, hidden_states, rotary, segments, key_valid, sequence_length):
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        query = apply_rotary(self.norm_q(query).to(value.dtype), rotary)
        key = apply_rotary(self.norm_k(key).to(value.dtype), rotary)
        parallel = get_parallel_state()
        if parallel.ulysses_enabled:
            # Exchange sequence for heads within the Ulysses group. Keep Q/K/V in
            # one collective so their backward exchanges stay in lockstep.
            qkv = torch.stack((query, key, value), dim=0).permute(3, 2, 0, 1, 4)
            qkv = gather_seq_scatter_heads(qkv, seq_dim=1, head_dim=0, group=parallel.ulysses_group)
            query, key, value = qkv.permute(2, 3, 1, 0, 4).unbind(0)
        prefix_length = segments[-1][1] if segments else 0
        outputs = []
        for start, end, is_text in [*segments, (prefix_length, key.shape[1], False)]:
            # Padded keys must never be attended to; padded query outputs are
            # removed after the final sequence gather.
            key_end = min(end, sequence_length)
            mask = None if key_valid is None else key_valid[:, None, None, :key_end]
            if is_text:
                queries = torch.arange(start, end, device=query.device)
                keys = torch.arange(key_end, device=query.device)
                causal = (keys[None, :] <= queries[:, None])[None, None]
                mask = causal if mask is None else mask & causal
            outputs.append(self._attend(query[:, start:end], key[:, :key_end], value[:, :key_end], mask))
        output = torch.cat(outputs, dim=1)
        if parallel.ulysses_enabled:
            output = gather_heads_scatter_seq(
                output.permute(2, 1, 0, 3), seq_dim=1, head_dim=0, group=parallel.ulysses_group
            ).permute(2, 1, 0, 3)
        return self.to_out[1](self.to_out[0](output.flatten(2).to(query.dtype)))


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, eps=config.eps, elementwise_affine=False)
        self.attn = QwenImage21Attention(config, dim)
        self.img_norm2 = nn.LayerNorm(dim, eps=config.eps, elementwise_affine=False)
        self.img_mlp = QwenImage21SwiGLU(dim, dim * config.mlp_ratio)

    def forward(self, hidden_states, modulation, rotary, target_mask, segments, key_valid, sequence_length):
        scale_attn, gate_attn, scale_mlp, gate_mlp = (
            select_modulation(chunk, target_mask) for chunk in modulation.chunk(4, dim=-1)
        )
        attention_input = self.img_norm1(hidden_states) * (1 + scale_attn)
        hidden_states = hidden_states + gate_attn.tanh() * self.attn(
            attention_input, rotary, segments, key_valid, sequence_length
        )
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
        parallel = get_parallel_state()
        if config.num_attention_heads % parallel.ulysses_size:
            raise ValueError("Qwen-Image 2.1 attention heads must be divisible by ulysses_size.")
        if parallel.ulysses_enabled and parallel.async_enabled:
            raise NotImplementedError("Qwen-Image 2.1 uses synchronous Ulysses. Set enable_async=false.")
        if parallel.ulysses_enabled and parallel.dp_mode == "eager":
            raise ValueError("Qwen-Image 2.1 sequence parallelism requires FSDP2 or DDP gradient reduction.")
        attention = kwargs.pop("attn_implementation", None)
        if attention is not None:
            config._attn_implementation = attention
        kwargs.pop("torch_dtype", None)
        kwargs.pop("trust_remote_code", None)
        super().__init__(config, **kwargs)
        self.config.tie_word_embeddings = False
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

    @staticmethod
    def get_flops_info(batch):
        """Return logical matmul layouts from prepared condition batches, before SP padding."""
        layouts = []
        for images, context, shapes, mask in zip(
            batch["hidden_states"], batch["encoder_hidden_states"], batch["img_shapes"], batch["img_mask"]
        ):
            lengths = [prod(shape) for shape in shapes]
            slots = mask[0].tolist()
            tokens = list(chain.from_iterable(repeat(value, 4 if value else 1) for value in slots))
            segments = _prefix_segments(tokens, lengths)
            sequence = len(tokens)
            pairs = sum((end - start) * end for start, end, _ in segments) + lengths[-1] * sequence
            layouts.append((images.shape[0], images.shape[1], context.shape[1], sequence, pairs))
        return layouts

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
            _, height, width = self._normalize_img_shapes(shapes)[0][-1]
            prediction_bchw = prediction.transpose(1, 2).unflatten(2, (height, width)).float()
            target_bchw = target.transpose(1, 2).unflatten(2, (height, width)).float()
            loss = torch.stack(
                [F.mse_loss(pred, truth) for pred, truth in zip(prediction_bchw.split(1), target_bchw.split(1))]
            )
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
