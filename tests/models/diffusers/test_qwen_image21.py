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

"""Strict checkpoint and upstream numerical contracts for Qwen-Image 2.1."""

from __future__ import annotations

from math import prod

import pytest
import torch
import torch.nn.functional as F
from diffusers import QwenImage21Transformer2DModel as OfficialTransformer

from tests.models.compare import assert_outputs_and_grads_match
from tests.models.tiny_configs import tiny_qwen_image21_config
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.diffusers.qwen_image21.qwen_image21_transformer.modeling_qwen_image21_transformer import (
    QwenImage21Transformer2DModel,
)


def tiny_config(attention="sdpa"):
    config = tiny_qwen_image21_config()
    config._set_ops_implementation(tiny_ops(attention))
    return config


def tiny_ops(attention="sdpa"):
    return OpsImplementationConfig(
        attn_implementation=attention,
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
    )


def sample_inputs(edit=False, requires_grad=True):
    """Non-square target and two adjacent reference-image spans exercise RoPE and masks."""
    if edit:
        # The adjacent three image slots represent TWO distinct reference images.
        # Collapsing them into one image run produces the wrong causal attention.
        shapes = [(1, 2, 2), (1, 2, 4), (1, 2, 4)]
        prompt_mask = [False, True, True, True, False, False]
    else:
        shapes = [(1, 2, 4)]
        prompt_mask = [False, False, False]
    text_length = len(prompt_mask)
    target_slots = prod(shapes[-1]) // 4
    return {
        "hidden_states": torch.randn(1, sum(prod(shape) for shape in shapes), 8, requires_grad=requires_grad),
        "encoder_hidden_states": torch.randn(1, text_length, 16, requires_grad=requires_grad),
        "encoder_hidden_states_mask": torch.tensor([[True] * (text_length - 1) + [False]]),
        "timestep": torch.tensor([0.37]),
        "img_shapes": [shapes],
        "img_mask": torch.tensor([prompt_mask + [True] * target_slots]),
    }


def clone_sample(sample):
    return {
        key: value.detach().clone().requires_grad_(value.requires_grad) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("edit", [False, True], ids=["text-to-image", "two-reference-edit"])
@pytest.mark.parametrize("training", [False, True], ids=["prediction", "supervised"])
def test_upstream_predictions_input_and_parameter_gradients(attention, edit, training):
    torch.manual_seed(41)
    config = tiny_config(attention)
    official = OfficialTransformer(**config.to_diffuser_dict())
    ours = QwenImage21Transformer2DModel(config)
    original_state, candidate_state = official.state_dict(), ours.state_dict()
    assert original_state.keys() == candidate_state.keys()
    assert {key: tensor.shape for key, tensor in original_state.items()} == {
        key: tensor.shape for key, tensor in candidate_state.items()
    }
    incompatible = ours.load_state_dict(original_state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    original_inputs = sample_inputs(edit)
    candidate_inputs = clone_sample(original_inputs)
    target_length = prod(original_inputs["img_shapes"][0][-1])
    target = torch.randn(1, target_length, 8)
    expected_prediction = None

    def call(model):
        nonlocal expected_prediction
        if model is official:
            # Upstream returns the joint sequence; the training adapter supervises
            # only the target image at its tail, excluding both reference images.
            expected_prediction = model(**original_inputs, return_dict=False)[0][:, -target_length:]
            return F.mse_loss(expected_prediction, target) if training else expected_prediction
        if training:
            output = model(
                **{key: [value] for key, value in candidate_inputs.items()},
                training_target=[target],
            )
            assert set(output.loss) == {"mse_loss"}
            assert len(output.predictions) == 1
            torch.testing.assert_close(output.predictions[0], expected_prediction, atol=2e-6, rtol=1e-5)
            return output.loss["mse_loss"]
        return model(**candidate_inputs, return_dict=False)[0]

    assert_outputs_and_grads_match(official, ours, call, atol=2e-6, rtol=1e-5, grad_atol=3e-6, grad_rtol=2e-5)
    for key in ("hidden_states", "encoder_hidden_states"):
        assert original_inputs[key].grad is not None
        torch.testing.assert_close(candidate_inputs[key].grad, original_inputs[key].grad, atol=3e-6, rtol=2e-5)
    # The last prompt token is padding and cannot affect the target prediction.
    assert torch.count_nonzero(candidate_inputs["encoder_hidden_states"].grad[:, -1]) == 0


def test_ragged_batch_averages_target_losses_and_gradients():
    torch.manual_seed(9)
    model = QwenImage21Transformer2DModel(tiny_config())
    samples = [sample_inputs(False), sample_inputs(True)]
    targets = [torch.randn(1, prod(sample["img_shapes"][0][-1]), 8) for sample in samples]
    reference = QwenImage21Transformer2DModel(tiny_config())
    reference.load_state_dict(model.state_dict(), strict=True)
    batch = {key: [sample[key] for sample in samples] for key in samples[0]}
    actual = model(**batch, training_target=targets)
    expected = torch.stack(
        [F.mse_loss(reference(**sample, return_dict=False)[0], target) for sample, target in zip(samples, targets)]
    ).mean()
    torch.testing.assert_close(actual.loss["mse_loss"], expected)
    actual.loss["mse_loss"].backward()
    expected.backward()
    for (name, parameter), (reference_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        if reference_parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, msg=name)


def test_checkpoint_roundtrip_preserves_upstream_names_and_attention(tmp_path):
    torch.manual_seed(7)
    config = tiny_config()
    official = OfficialTransformer(**config.to_diffuser_dict())
    model = QwenImage21Transformer2DModel(config).eval()
    model.load_state_dict(official.state_dict(), strict=True)
    model.save_pretrained(tmp_path)
    reloaded = QwenImage21Transformer2DModel.from_pretrained(tmp_path, ops_implementation=tiny_ops()).eval()
    assert reloaded.config.model_type == "qwen_image21_transformer"
    assert reloaded.state_dict().keys() == official.state_dict().keys()
    for key, expected in official.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], expected, atol=0, rtol=0)
    sample = sample_inputs(True, requires_grad=False)
    with torch.no_grad():
        expected = model(**sample, return_dict=False)[0]
        actual = reloaded(**sample, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_masked_first_prompt_query_has_finite_upstream_gradients():
    torch.manual_seed(11)
    config = tiny_config("eager")
    official = OfficialTransformer(**config.to_diffuser_dict())
    ours = QwenImage21Transformer2DModel(config)
    ours.load_state_dict(official.state_dict(), strict=True)
    sample = sample_inputs(False, requires_grad=False)
    # The first causal text query has no valid keys. Its zero attention row
    # must remain finite in backward, not just be patched after softmax.
    sample["encoder_hidden_states_mask"][:, 0] = False

    def call(model):
        return model(**sample, return_dict=False)[0][:, -8:]

    assert_outputs_and_grads_match(official, ours, call, atol=2e-6, rtol=1e-5, grad_atol=3e-6, grad_rtol=2e-5)
    assert all(torch.isfinite(parameter.grad).all() for parameter in ours.parameters() if parameter.grad is not None)
