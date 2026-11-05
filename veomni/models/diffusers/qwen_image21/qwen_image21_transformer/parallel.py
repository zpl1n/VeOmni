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

"""Joint-sequence parallel execution for Qwen-Image 2.1 training.

Ulysses uses head/sequence all-to-all; CP keeps queries local and gathers K/V.
The CP baseline is intentionally not a ring-attention implementation.
"""

from math import prod

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_qwenimage21 import _qwenimage21_prefix_segments

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    slice_input_tensor,
)


def parallel_attention(attention, attn, query, key, value, segments, key_valid):
    """Run exact block-causal attention after the topology's Q/K/V exchange."""
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
        mask = key_valid[:, None, None, :end]
        if is_text:
            queries = torch.arange(left, right, device=query.device)
            keys = torch.arange(end, device=query.device)
            mask = mask & (keys[None, :] <= queries[:, None])[None, None]
        outputs.append(
            attention(attn, query[:, left - query_start : right - query_start], key[:, :end], value[:, :end], mask)
        )
    output = torch.cat(outputs, dim=1)
    if state.ulysses_enabled:
        output = gather_heads_scatter_seq(
            output.permute(2, 1, 0, 3), seq_dim=1, head_dim=0, group=state.ulysses_group
        ).permute(2, 1, 0, 3)
    return output


def parallel_forward(model, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask, text_mask):
    """Shard the interleaved joint sequence before all Transformer blocks."""
    state = get_parallel_state()
    hidden_states = model.img_in(hidden_states)
    context = model.txt_in(encoder_hidden_states)
    repeats = torch.where(img_mask, 4, 1)[0]
    image_mask = torch.repeat_interleave(img_mask[0], repeats)
    target_tokens = prod(img_shapes[0][-1])
    joint = torch.cat([context, context.new_zeros(context.shape[0], target_tokens // 4, context.shape[2])], dim=1)
    joint = joint.repeat_interleave(repeats, dim=1)
    joint[:, image_mask] = hidden_states
    length = joint.shape[1]
    image_ids, target_mask = model.build_token_metadata(image_mask, img_shapes[0])
    prefix_length = length - target_tokens
    segments = _qwenimage21_prefix_segments(image_ids, prefix_length)
    rotary = model.pos_embed(img_shapes[0], image_mask, device=hidden_states.device)
    if rotary.is_complex():
        rotary = torch.view_as_real(rotary)
    timestep = timestep.to(hidden_states.dtype)
    modulation_mask = None
    if model.config.causal_condition:
        timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        modulation_mask = target_mask
    temb = model.time_text_embed(timestep, hidden_states)
    modulation = model.modulation(temb)
    key_valid = torch.ones(context.shape[0], length, dtype=torch.bool, device=joint.device)
    if text_mask is not None:
        text_positions = (~image_mask).nonzero(as_tuple=True)[0]
        vlm_text = ~img_mask[0][: text_mask.shape[1]]
        key_valid[:, text_positions] = text_mask.bool()[:, vlm_text]
    padding = (-length) % state.sp_size
    joint = F.pad(joint, (0, 0, 0, padding))
    rotary = F.pad(rotary, (0, 0, 0, 0, 0, padding))
    key_valid = F.pad(key_valid, (0, padding), value=False)
    joint = slice_input_tensor(joint, dim=1, padding=False)
    rotary = slice_input_tensor(rotary, dim=0, padding=False)
    if modulation_mask is not None:
        modulation_mask = slice_input_tensor(F.pad(modulation_mask, (0, padding), value=False), dim=0, padding=False)
    for block in model.transformer_blocks:
        args = (joint, modulation, rotary, None, modulation_mask, None, None, None, segments, key_valid)
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            joint = model._gradient_checkpointing_func(block, *args)
        else:
            joint = block(*args)
    output = model.proj_out(model.norm_out(joint, temb, modulation_mask))
    # The full loss is replicated on SP ranks. Gather backward sums its copies,
    # compensating FSDP2's averaging over the combined DP/SP mesh.
    output = gather_outputs(output, gather_dim=1)
    return output[:, :length][:, -target_tokens:]
