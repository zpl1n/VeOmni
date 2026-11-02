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

"""Two-rank worker comparing a tiny DiT's actual accelerator and FSDP2 training math."""

from __future__ import annotations

import json
import os
import sys
from math import prod
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from diffusers import QwenImage21Transformer2DModel as OfficialTransformer
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from tests.models.diffusers.test_qwen_image21 import sample_inputs, tiny_config, tiny_ops
from veomni.arguments.arguments_types import AcceleratorConfig, MixedPrecisionConfig
from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config, use_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.models.diffusers.qwen_image21.qwen_image21_transformer.modeling_qwen_image21_transformer import (
    QwenImage21Transformer2DModel,
)
from veomni.utils.device import get_device_type, get_dist_comm_backend, set_device


def tree_to(value, device):
    if torch.is_tensor(value):
        return value.detach().clone().to(device)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to(item, device) for item in value]
    return value


def full_cpu(value):
    if isinstance(value, DTensor):
        value = value.full_tensor()
    return value.detach().cpu()


def reference_forward(model, samples, targets):
    predictions = [
        model(**sample, return_dict=False)[0][:, -prod(sample["img_shapes"][0][-1]) :] for sample in samples
    ]
    loss = torch.stack([F.mse_loss(prediction, target) for prediction, target in zip(predictions, targets)]).mean()
    return predictions, loss


def candidate_forward(model, samples, targets):
    result = model(**{key: [sample[key] for sample in samples] for key in samples[0]}, training_target=targets)
    return result.predictions, result.loss["mse_loss"]


def main(output_dir, mode):
    if mode not in ("fp32", "bf16-gc"):
        raise ValueError(f"Unknown precision mode: {mode}")
    bf16 = mode == "bf16-gc"
    device = get_device_type()
    if device not in ("cuda", "npu"):
        raise RuntimeError("This worker requires real accelerator execution; no CPU fallback is allowed.")
    set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    torch.distributed.init_process_group(get_dist_comm_backend())
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size != 2:
        raise ValueError("The FSDP2 regression uses exactly two ranks.")
    init_parallel_state_from_config(AcceleratorConfig(dp_replicate_size=1, dp_shard_size=world_size), name="base")
    try:
        with use_parallel_state("base"):
            torch.manual_seed(427)
            config = tiny_config()
            reference = OfficialTransformer(**config.to_diffuser_dict()).eval()
            unsharded = QwenImage21Transformer2DModel(tiny_config()).train()
            unsharded.load_state_dict(reference.state_dict(), strict=True)
            checkpoint = Path(output_dir) / "model"
            if rank == 0:
                unsharded.save_pretrained(checkpoint)
            torch.distributed.barrier()
            unsharded.to(device)
            if bf16:
                unsharded.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                # A singleton mesh matches FSDP's FP32 master weights and BF16
                # computation on one device, without cross-rank parameter sharding.
                singleton_mesh = DeviceMesh(
                    device, torch.arange(world_size).reshape(world_size, 1), mesh_dim_names=("replica", "single")
                )["single"]
                policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
                )
                for block in unsharded.transformer_blocks:
                    fully_shard(block, mesh=singleton_mesh, mp_policy=policy)
                fully_shard(unsharded, mesh=singleton_mesh, mp_policy=policy)
            sharded = build_foundation_model(
                str(checkpoint),
                init_device="meta",
                torch_dtype="float32",
                ops_implementation=tiny_ops(),
            )
            sharded = build_parallelize_model(
                sharded,
                weights_path=str(checkpoint),
                init_device="meta",
                mixed_precision=MixedPrecisionConfig(enable=bf16, cast_forward_inputs=False),
                enable_gradient_checkpointing=bf16,
                enable_reshard_after_forward=True,
            ).train()
            if bf16:
                assert unsharded.gradient_checkpointing and sharded.gradient_checkpointing
            assert any(isinstance(parameter, DTensor) for parameter in sharded.parameters())
            cpu_samples = [sample_inputs(False, requires_grad=False), sample_inputs(True, requires_grad=False)]
            cpu_targets = [torch.randn(1, prod(sample["img_shapes"][0][-1]), 8) for sample in cpu_samples]
            samples, targets = tree_to(cpu_samples, device), tree_to(cpu_targets, device)
            # A BF16 model is not an elementwise FP32 reference: even upstream
            # CPU BF16 exceeds that bound. The FP32 case checks upstream math;
            # BF16 checks the default precision/GC against the same-precision
            # singleton mesh, including FP32 master weights and updates.
            models = (unsharded, sharded) if bf16 else (reference, unsharded, sharded)
            plain_parameters = dict(unsharded.named_parameters())
            for name, parameter in sharded.named_parameters():
                torch.testing.assert_close(full_cpu(parameter), full_cpu(plain_parameters[name]), atol=0, rtol=0)
            # SGD keeps the comparison well-conditioned even for near-zero gradients.
            # The trainer integration separately exercises VeOmni's default AdamW.
            optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in models]
            for step in range(2):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                # Diffusers selects fused RMSNorm by package availability on some versions;
                # a CPU reference must always use CPU math, even on an Ascend host.
                if not bf16:
                    with patch("diffusers.models.normalization.is_torch_npu_available", return_value=False):
                        cpu_predictions, cpu_loss = reference_forward(reference, cpu_samples, cpu_targets)
                        cpu_loss.backward()
                predictions, loss = candidate_forward(unsharded, samples, targets)
                sharded_predictions, sharded_loss = candidate_forward(sharded, samples, targets)
                if not bf16:
                    torch.testing.assert_close(loss.cpu(), cpu_loss, atol=3e-5, rtol=3e-4)
                    for cpu, plain in zip(cpu_predictions, predictions, strict=True):
                        torch.testing.assert_close(plain.cpu(), cpu, atol=3e-5, rtol=3e-4)
                torch.testing.assert_close(sharded_loss, loss, atol=2e-5, rtol=2e-4)
                for plain, fsdp in zip(predictions, sharded_predictions, strict=True):
                    torch.testing.assert_close(fsdp, plain, atol=2e-5, rtol=2e-4)
                loss.backward()
                sharded_loss.backward()
                cpu_parameters = dict(reference.named_parameters())
                plain_parameters = dict(unsharded.named_parameters())
                sharded_parameters = dict(sharded.named_parameters())
                assert cpu_parameters.keys() == plain_parameters.keys() == sharded_parameters.keys()
                nonzero_gradients = 0
                for name, parameter in plain_parameters.items():
                    actual = sharded_parameters[name].grad
                    if parameter.grad is None:
                        assert actual is None, name
                        if not bf16:
                            assert cpu_parameters[name].grad is None, name
                        continue
                    assert actual is not None, name
                    plain_grad, fsdp_grad = full_cpu(parameter.grad), full_cpu(actual)
                    assert torch.isfinite(plain_grad).all() and torch.isfinite(fsdp_grad).all(), name
                    nonzero_gradients += int(torch.count_nonzero(plain_grad))
                    if not bf16:
                        expected = cpu_parameters[name].grad
                        assert expected is not None, name
                        torch.testing.assert_close(plain_grad, expected, atol=5e-5, rtol=5e-4, msg=name)
                    torch.testing.assert_close(fsdp_grad, plain_grad, atol=2e-5, rtol=2e-4, msg=name)
                assert nonzero_gradients > 0
                before = {name: full_cpu(parameter).clone() for name, parameter in plain_parameters.items()}
                for optimizer in optimizers:
                    optimizer.step()
                changed = False
                for name, parameter in plain_parameters.items():
                    plain, fsdp = full_cpu(parameter), full_cpu(sharded_parameters[name])
                    changed |= not torch.equal(plain, before[name])
                    torch.testing.assert_close(fsdp, plain, atol=2e-5, rtol=2e-4, msg=f"step {step}: {name}")
                assert changed
            if rank == 0:
                (Path(output_dir) / "result.json").write_text(
                    json.dumps(
                        {
                            "device": device,
                            "ranks": world_size,
                            "steps": 2,
                            "mode": mode,
                            "upstream_cpu": "covered-by-fp32-case" if bf16 else "PASS",
                            "fsdp2": "PASS",
                        }
                    )
                )
    finally:
        torch.distributed.destroy_process_group()
        clear_parallel_state()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
