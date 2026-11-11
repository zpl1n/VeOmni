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

"""Model-local FLOPs accounting and training-log integration contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.diffusers.test_qwen_image21 import sample_inputs, tiny_config
from veomni.models.diffusers.qwen_image21 import mfu as mfu_module
from veomni.models.diffusers.qwen_image21.flops import QwenImage21FlopsCounter
from veomni.models.diffusers.qwen_image21.qwen_image21_transformer.modeling_qwen_image21_transformer import (
    QwenImage21Transformer2DModel,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def cpu_checkpoint_device():
    previous = torch.utils.checkpoint.DefaultDeviceType.get_device_type()
    torch.utils.checkpoint.DefaultDeviceType.set_device_type("cpu")
    yield
    torch.utils.checkpoint.DefaultDeviceType.set_device_type(previous)


@pytest.mark.parametrize("edit,batch_size", [(False, 1), (True, 2)])
@pytest.mark.parametrize("causal_condition", [False, True])
def test_counter_matches_executed_matmul_shapes(monkeypatch, edit, batch_size, causal_condition):
    """Derive the reference from executed layer shapes rather than its formula."""
    config = tiny_config("eager")
    config.causal_condition = causal_condition
    model = QwenImage21Transformer2DModel(config).train()
    if edit:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    counter = QwenImage21FlopsCounter(config)
    observed = [0]

    def count_linear(module, inputs, _output):
        observed[0] += 6 * inputs[0].numel() * module.out_features

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(count_linear)
    for block in model.transformer_blocks:
        original = block.attn._attend

        def count_attention(query, key, value, mask, original=original):
            batch, queries, heads, width = query.shape
            observed[0] += 12 * batch * queries * key.shape[1] * heads * width
            return original(query, key, value, mask)

        monkeypatch.setattr(block.attn, "_attend", count_attention)

    sample = sample_inputs(edit, requires_grad=False)
    for key, value in sample.items():
        if torch.is_tensor(value):
            sample[key] = value.expand(batch_size, *value.shape[1:]).clone()
    sample["img_shapes"] *= batch_size
    joint_tokens = sample["hidden_states"].shape[1] + int((~sample["img_mask"][0]).sum())
    baseline = model(**sample, return_dict=False)[0]
    forward_flops = observed[0]
    baseline.square().mean().backward()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    observed[0] = 0
    model._mfu_flops_counter = counter
    for micro_step in range(1, 3):
        actual = model(**sample, return_dict=False)[0]
        torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        actual.square().mean().backward()
        assert counter.total_flops == micro_step * forward_flops
    for name, parameter in model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(parameter.grad, 2 * expected_gradients[name], atol=0, rtol=0, msg=name)
        else:
            assert parameter.grad is None
    assert counter.total_tokens == 2 * batch_size * joint_tokens
    counter.reset()
    assert counter.total_flops == counter.total_tokens == 0


@pytest.mark.parametrize("peak_tflops", [354.0, None])
def test_step_accumulation_dp_reduction_and_logged_mfu(monkeypatch, peak_tflops):
    dp_group, fsdp_group = object(), object()
    parallel = SimpleNamespace(dp_group=dp_group, fsdp_group=fsdp_group, world_size=8)
    monkeypatch.setattr("veomni.trainer.callbacks.base.get_parallel_state", lambda: parallel)
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(mfu_module.time, "perf_counter", lambda: next(clock))
    synchronize = Mock()
    monkeypatch.setattr(mfu_module, "synchronize", synchronize)
    logger = Mock()
    monkeypatch.setattr(mfu_module, "logger", logger)
    trainer = SimpleNamespace(
        model=SimpleNamespace(unwrapped_module=SimpleNamespace(), model_config=tiny_config()),
        args=SimpleNamespace(train=SimpleNamespace(global_rank=0)),
        environ_meter=SimpleNamespace(consume_tokens=10),
    )

    def original_step_end(_state, **_kwargs):
        trainer.environ_meter.consume_tokens += 20
        trainer.step_env_metrics = {"mfu": 0.0, "flops_promised(T)": float("inf")}
        trainer.step_train_metrics = {"training/total_loss": 2.0}

    meter = Mock()
    meter.on_step_end.side_effect = original_step_end

    def reduce(data, op, group):
        if op == "sum":
            assert group is dp_group
            # Eight devices with SP4 have two distinct data-parallel samples.
            return [value * 2 for value in data]
        assert op == "max" and group is fsdp_group
        return 3.0

    monkeypatch.setattr(mfu_module, "all_reduce", reduce)
    callback = mfu_module.QwenImage21MFUCallback(trainer, meter, peak_tflops)
    callback.counter.total_flops = 123
    state, micro_batches = SimpleNamespace(global_step=7), [{"sample": 1}]
    callback.on_step_begin(state, micro_batches=micro_batches)
    assert callback.counter.total_flops == callback.counter.total_tokens == 0
    assert callback.module._mfu_flops_counter is callback.counter
    callback.counter.add(1, 8192, 8, 8200, [(0, 8, True)])
    callback.counter.add(2, 4096, 4, 4100, [(0, 4, True)])
    local_flops, local_tokens = callback.counter.total_flops, callback.counter.total_tokens
    callback.on_step_end(state, loss=2.0, loss_dict={"mse_loss": 2.0}, grad_norm=1.0)

    meter.on_step_begin.assert_called_once_with(state, micro_batches=micro_batches)
    meter.on_step_end.assert_called_once_with(state, loss=2.0, loss_dict={"mse_loss": 2.0}, grad_norm=1.0)
    assert trainer.environ_meter.consume_tokens == 30
    assert callback.module._mfu_flops_counter is None
    assert synchronize.call_count == 2
    env, training = trainer.step_env_metrics, trainer.step_train_metrics
    achieved = local_flops * 2 / 3.0 / 1e12
    assert env["flops_achieved(T)"] == pytest.approx(achieved, rel=1e-12, abs=0)
    assert env["mfu/joint_tokens"] == local_tokens * 2
    assert env["mfu/step_time"] == 3.0
    assert training["training/dit_tflops_per_device"] == pytest.approx(achieved / 8)
    assert env["training/total_loss"] == 2.0
    if peak_tflops is not None:
        expected = achieved / (8 * peak_tflops)
        assert env["mfu"] == pytest.approx(expected, rel=1e-12, abs=0)
        assert env["flops_promised(T)"] == 8 * peak_tflops
        assert training["training/mfu(%)"] == pytest.approx(100 * expected)
    else:
        assert "mfu" not in env and "flops_promised(T)" not in env
        assert "training/mfu(%)" not in training
        assert "unavailable" in logger.info.call_args[0][0]


@pytest.mark.parametrize("peak", [0, -1, float("inf"), float("nan")])
def test_peak_tflops_rejects_unusable_hardware_denominators(peak):
    with pytest.raises(ValueError, match="finite and positive"):
        mfu_module.QwenImage21MFUConfig(peak_tflops=peak)


@pytest.mark.parametrize(
    "mixed_precision,detected,explicit,expected,queries",
    [
        (True, 354.0, None, 354.0, 1),
        (True, float("inf"), None, None, 1),
        (False, 354.0, None, None, 0),
        (True, float("inf"), 123.0, 123.0, 0),
    ],
)
def test_trainer_replaces_only_meter_and_resolves_compute_peak(
    monkeypatch, mixed_precision, detected, explicit, expected, queries
):
    config = tiny_config("eager")
    module = QwenImage21Transformer2DModel(config)
    assert module.dtype == torch.float32
    mp = SimpleNamespace(enable=mixed_precision, param_dtype="bfloat16")
    args = SimpleNamespace(
        mfu=mfu_module.QwenImage21MFUConfig(peak_tflops=explicit),
        train=SimpleNamespace(training_task="online_training", global_rank=1),
        model=SimpleNamespace(
            accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_mode="fsdp2", mixed_precision=mp))
        ),
    )
    before, meter, after = object(), Mock(), object()
    base = SimpleNamespace(
        args=args,
        model=SimpleNamespace(model_config=config, unwrapped_module=module),
        environ_meter_callback=meter,
        _callbacks=[before, meter, after],
    )
    monkeypatch.setattr(mfu_module.DiTTrainer, "__init__", lambda self, _args: setattr(self, "base", base))
    monkeypatch.setattr("veomni.trainer.callbacks.base.get_parallel_state", lambda: SimpleNamespace(world_size=2))
    device_peak = Mock(return_value=detected)
    monkeypatch.setattr(mfu_module, "get_device_flops", device_peak)

    trainer = mfu_module.QwenImage21MFUTrainer(args)

    callback = trainer.base.environ_meter_callback
    assert isinstance(callback, mfu_module.QwenImage21MFUCallback)
    assert trainer.base._callbacks == [before, callback, after]
    assert callback.meter_callback is meter and callback.module is module
    assert callback.peak_tflops == expected
    assert device_peak.call_count == queries
