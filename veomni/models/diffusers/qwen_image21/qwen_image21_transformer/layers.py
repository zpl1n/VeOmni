# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Single-stream Qwen-Image 2.1 layers following the DiffSynth-Studio DiT.

Parameter names match the released Diffusers checkpoint. Rotary frequencies
use real pairs so the same implementation can run on Ascend.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


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
            self.frequencies.append(torch.stack((angles.cos(), angles.sin()), dim=-1))

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
    real, imaginary = hidden_states.float().unflatten(-1, (-1, 2)).unbind(-1)
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
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        if self.zero_centered:
            return (normalized * (1 + self.weight.float())).to(hidden_states.dtype)
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
