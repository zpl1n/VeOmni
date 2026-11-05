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

"""CPU/Gloo numerical reference for Qwen-Image 2.1 SP and CP (no accelerator)."""

import json
import sys
from math import prod
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from diffusers import QwenImage21Transformer2DModel as OfficialTransformer

from tests.models.diffusers.test_qwen_image21 import sample_inputs, tiny_config
from veomni.arguments.arguments_types import AcceleratorConfig
from veomni.distributed import parallel_state
from veomni.models.diffusers.qwen_image21.qwen_image21_transformer.modeling_qwen_image21_transformer import (
    QwenImage21Transformer2DModel,
)


def mean_world(tensor):
    tensor = tensor.detach().clone()
    dist.all_reduce(tensor)
    return tensor / dist.get_world_size()


def main(output_dir):
    torch.set_num_threads(1)
    torch.utils.checkpoint.DefaultDeviceType.set_device_type("cpu")
    dist.init_process_group("gloo")
    world = dist.get_world_size()
    try:
        with (
            patch.object(parallel_state, "get_device_type", return_value="cpu"),
            patch("diffusers.models.normalization.is_torch_npu_available", return_value=False),
        ):
            modes = [("sp2", 2, 1), ("sp4", 4, 1), ("cp2", 1, 2), ("cp4", 1, 4)]
            for name, sp, cp in modes:
                config = AcceleratorConfig(
                    dp_replicate_size=1, dp_shard_size=world // (sp * cp), ulysses_size=sp, cp_size=cp
                )
                parallel_state.init_parallel_state_from_config(config, name=name)
            parallel_state.init_parallel_state_from_config(
                AcceleratorConfig(dp_replicate_size=1, dp_shard_size=world), name="reference"
            )
            reports = []
            for name, _, _ in modes:
                for checkpointing in (False, True):
                    with parallel_state.use_parallel_state(name):
                        torch.manual_seed(72)
                        config = tiny_config()
                        config.num_attention_heads = 4
                        model = QwenImage21Transformer2DModel(config).train()
                        if checkpointing:
                            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        reference = OfficialTransformer(**model.config.to_diffuser_dict()).train()
                        reference.load_state_dict(model.state_dict(), strict=True)
                        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
                        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
                        for step in range(2):
                            optimizer.zero_grad(set_to_none=True)
                            ref_optimizer.zero_grad(set_to_none=True)
                            # DP groups use different samples; SP peers share one.
                            torch.manual_seed(80 + step + parallel_state.get_parallel_state().dp_rank)
                            samples = [sample_inputs(edit, requires_grad=False) for edit in (False, True)]
                            targets = [torch.randn(1, prod(sample["img_shapes"][0][-1]), 8) for sample in samples]
                            reference_predictions = [
                                reference(**sample, return_dict=False)[0][:, -target.shape[1] :]
                                for sample, target in zip(samples, targets)
                            ]
                            weights = [torch.tensor([0.7]), torch.tensor([1.3])]
                            ref_loss = torch.stack(
                                [
                                    torch.nn.functional.mse_loss(prediction.float(), target.float()) * weight.squeeze()
                                    for prediction, target, weight in zip(reference_predictions, targets, weights)
                                ]
                            ).mean()
                            result = model(
                                **{key: [sample[key] for sample in samples] for key in samples[0]},
                                training_target=targets,
                                loss_weight=weights,
                            )
                            torch.testing.assert_close(result.loss["mse_loss"], ref_loss, atol=3e-6, rtol=3e-5)
                            for actual, expected in zip(result.predictions, reference_predictions):
                                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                            ref_loss.backward()
                            result.loss["mse_loss"].backward()
                            for (key, parameter), (ref_key, ref_parameter) in zip(
                                model.named_parameters(), reference.named_parameters()
                            ):
                                assert key == ref_key
                                assert parameter.grad is not None and ref_parameter.grad is not None, key
                                parameter.grad.copy_(mean_world(parameter.grad))
                                ref_parameter.grad.copy_(mean_world(ref_parameter.grad))
                                torch.testing.assert_close(
                                    parameter.grad, ref_parameter.grad, atol=5e-6, rtol=5e-5, msg=key
                                )
                            optimizer.step()
                            ref_optimizer.step()
                            for parameter, expected in zip(model.parameters(), reference.parameters()):
                                torch.testing.assert_close(parameter, expected, atol=3e-6, rtol=3e-5)
                        reports.append({"mode": name, "gradient_checkpointing": checkpointing, "steps": 2})
            if dist.get_rank() == 0:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                (Path(output_dir) / "result.json").write_text(json.dumps(reports))
                print("QWEN_IMAGE21_SP_CP_CPU_PASS", reports, flush=True)
    finally:
        dist.destroy_process_group()
        parallel_state.clear_parallel_state()


if __name__ == "__main__":
    main(sys.argv[1])
