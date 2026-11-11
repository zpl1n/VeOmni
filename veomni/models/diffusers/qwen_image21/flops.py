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

"""Analytical matmul FLOPs for Qwen-Image 2.1 full DiT training."""


class QwenImage21FlopsCounter:
    """Accumulate forward-plus-backward model FLOPs before SP padding.

    Use six FLOPs per linear weight use and twelve per attention width/pair,
    matching the framework's full-training convention. Dense attention rectangles
    include masked positions. Frozen encoders, checkpoint recomputation, padding,
    elementwise operations, and optimizer work are excluded.
    """

    def __init__(self, config):
        self.hidden_size = config.num_attention_heads * config.attention_head_dim
        self.num_layers = config.num_layers
        self.mlp_ratio = config.mlp_ratio
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels or config.in_channels
        self.context_in_dim = config.context_in_dim
        self.causal_condition = config.causal_condition
        self.reset()

    def reset(self):
        """Clear the accumulated FLOPs and unpadded joint-token count."""
        self.total_flops = 0
        self.total_tokens = 0

    def add(self, batch_size, image_tokens, text_tokens, joint_tokens, segments):
        """Count one prediction call with prefix ``(start, end, is_text)`` segments.

        ``text_tokens`` is the encoder-feature length before image-slot expansion.
        The target-image segment follows the supplied prefix segments.
        """
        dim = self.hidden_size
        prefix_end = segments[-1][1] if segments else 0
        attention_pairs = sum((end - start) * end for start, end, _ in segments)
        attention_pairs += (joint_tokens - prefix_end) * joint_tokens
        block_weights = self.num_layers * (4 + 3 * self.mlp_ratio) * dim**2
        linear_uses = batch_size * (
            image_tokens * self.in_channels * dim
            + text_tokens * (self.context_in_dim * dim + dim**2)
            + joint_tokens * (self.out_channels * dim + block_weights)
        )
        timestep_count = batch_size + int(self.causal_condition)
        linear_uses += timestep_count * (256 * dim + 6 * dim**2)
        self.total_flops += 6 * linear_uses + 12 * batch_size * self.num_layers * dim * attention_pairs
        self.total_tokens += batch_size * joint_tokens
