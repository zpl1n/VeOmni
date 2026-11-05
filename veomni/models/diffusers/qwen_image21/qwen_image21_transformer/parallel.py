# Copyright 2026 Qwen-Image Team, The HuggingFace Team, and Bytedance Ltd.
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

"""Joint-sequence execution for Qwen-Image 2.1 training.

Ulysses uses head/sequence all-to-all; CP keeps queries local and gathers K/V.
The CP baseline is intentionally not a ring-attention implementation.
"""

from itertools import chain, groupby, repeat
from math import prod

import torch
import torch.nn.functional as F

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    slice_input_tensor,
)


_LATENT_TOKENS_PER_IMAGE_SLOT = 4


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


def joint_attention(attention, attn, query, key, value, segments, key_valid):
    """Apply the same block-causal rule in serial, Ulysses, and CP."""
    state = get_parallel_state()
    if state.ulysses_enabled:
        query, key, value = [
            gather_seq_scatter_heads(
                tensor.permute(2, 1, 0, 3), seq_dim=1, head_dim=0, group=state.ulysses_group
            ).permute(2, 1, 0, 3)
            for tensor in (query, key, value)
        ]
    elif state.cp_enabled:
        # Keys/values are used by different local query rows, so backward must
        # SUM all ranks' contributions before returning each source shard.
        key = gather_outputs(key, gather_dim=1, group=state.cp_group)
        value = gather_outputs(value, gather_dim=1, group=state.cp_group)
    query_start = state.cp_rank * query.shape[1] if state.cp_enabled else 0
    query_end = query_start + query.shape[1]
    prefix_length = segments[-1][1] if segments else 0
    ranges = [*segments, (prefix_length, key.shape[1], False)]
    outputs = []
    for start, end, is_text in ranges:
        left, right = max(start, query_start), min(end, query_end)
        if left >= right:
            continue
        mask = None if key_valid is None else key_valid[:, None, None, :end]
        if is_text:
            queries = torch.arange(left, right, device=query.device)
            keys = torch.arange(end, device=query.device)
            causal = (keys[None, :] <= queries[:, None])[None, None]
            mask = causal if mask is None else mask & causal
        outputs.append(
            attention(attn, query[:, left - query_start : right - query_start], key[:, :end], value[:, :end], mask)
        )
    output = torch.cat(outputs, dim=1)
    if state.ulysses_enabled:
        output = gather_heads_scatter_seq(
            output.permute(2, 1, 0, 3), seq_dim=1, head_dim=0, group=state.ulysses_group
        ).permute(2, 1, 0, 3)
    return output


def joint_forward(model, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask, text_mask):
    """Interleave text and latents before running the optionally sharded blocks."""
    state = get_parallel_state()
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
    if text_mask is not None or state.sp_enabled:
        key_valid = torch.ones(context.shape[0], length, dtype=torch.bool, device=joint.device)
    if text_mask is not None:
        text_positions = (~image_mask).nonzero(as_tuple=True)[0]
        vlm_text = ~img_mask[0][: text_mask.shape[1]]
        key_valid[:, text_positions] = text_mask.bool()[:, vlm_text]
    if state.sp_enabled:
        padding = (-length) % state.sp_size
        joint = slice_input_tensor(F.pad(joint, (0, 0, 0, padding)), dim=1, padding=False)
        rotary = slice_input_tensor(F.pad(rotary, (0, 0, 0, 0, 0, padding)), dim=0, padding=False)
        key_valid = F.pad(key_valid, (0, padding), value=False)
        if modulation_mask is not None:
            modulation_mask = slice_input_tensor(
                F.pad(modulation_mask, (0, padding), value=False), dim=0, padding=False
            )
    for block in model.transformer_blocks:
        args = (joint, modulation, rotary, modulation_mask, segments, key_valid)
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            joint = model._gradient_checkpointing_func(block, *args)
        else:
            joint = block(*args)
    output = model.proj_out(model.norm_out(joint, temb, modulation_mask))
    if state.sp_enabled:
        # Gather backward sums the replicated loss gradients, compensating for
        # FSDP2's averaging over the combined DP/SP mesh.
        output = gather_outputs(output, gather_dim=1)
    return output[:, prefix_length:length]
