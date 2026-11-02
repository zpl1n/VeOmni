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

"""Tiny real Qwen3-VL/RGBA-VAE fixtures exercising the native DiT training path."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from scripts.ci.accelerator.launch_plan import CasePlan, LaunchSpec, runner_device_count
from tests.models.diffusers.test_qwen_image21 import tiny_config, tiny_ops
from veomni.arguments.arguments_types import AcceleratorConfig
from veomni.data import build_data_transform, build_dataloader, build_dataset
from veomni.distributed import parallel_state
from veomni.models import build_foundation_model
from veomni.models.diffusers.qwen_image21.qwen_image21 import modeling_qwen_image21
from veomni.models.diffusers.qwen_image21.qwen_image21.configuration_qwen_image21 import (
    QwenImage21ConditionModelConfig,
)
from veomni.models.diffusers.qwen_image21.qwen_image21.modeling_qwen_image21 import QwenImage21ConditionModel
from veomni.models.diffusers.qwen_image21.qwen_image21_transformer.modeling_qwen_image21_transformer import (
    QwenImage21Transformer2DModel,
)
from veomni.optim import build_optimizer
from veomni.trainer.dit_trainer import DiTDataCollator, DiTTrainer, OfflineEmbeddingSaver
from veomni.utils.device import get_device_type


ROOT = Path(__file__).resolve().parents[2]
_TWO_RANKS = LaunchSpec(nproc=2)


@pytest.fixture(scope="module")
def cpu_rank(tmp_path_factory):
    """Real one-rank state for the framework loader, with CPU numerical components."""
    previous_threads = torch.get_num_threads()
    rendezvous = tmp_path_factory.mktemp("qwen21_gloo") / "rendezvous"
    with pytest.MonkeyPatch.context() as patch, torch.random.fork_rng(devices=[]):
        patch.setattr(modeling_qwen_image21, "get_device_type", lambda: "cpu")
        patch.setattr(parallel_state, "get_device_type", lambda: "cpu")
        torch.set_num_threads(1)
        torch.distributed.init_process_group("gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1)
        parallel_state.init_parallel_state_from_config(AcceleratorConfig(), name="base")
        try:
            yield
        finally:
            torch.distributed.destroy_process_group()
            parallel_state.clear_parallel_state()
            torch.set_num_threads(previous_threads)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory, cpu_rank):
    """No downloads: real tiny encoder and VAE with the released 64-channel latent contract."""
    root = tmp_path_factory.mktemp("qwen21_components")
    torch.manual_seed(123)
    special = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    vocab = {token: index for index, token in enumerate(special + sorted(ByteLevel.alphabet()))}
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], pad_token=special[0], eos_token=special[2])
    tokenizer.add_special_tokens({"additional_special_tokens": special[1:]})
    # Match the official raw system/user/assistant template; no network tokenizer is used.
    template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for content in message['content'] %}"
        "{% if content['type'] == 'text' %}{{ content['text'] }}{% endif %}"
        "{% endfor %}{% endif %}{{ '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    tokenizer.chat_template = template
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2, min_pixels=32 * 32, max_pixels=64 * 64
        ),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        tokenizer=tokenizer,
        chat_template=template,
    )
    processor.save_pretrained(root / "processor")
    text = {
        "vocab_size": len(tokenizer),
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 512,
        "pad_token_id": 0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [1, 1, 2]},
    }
    vision = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "out_hidden_size": 16,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 64,
        "deepstack_visual_indexes": [0],
    }
    config = Qwen3VLConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    config._attn_implementation = "eager"
    Qwen3VLForConditionalGeneration(config).save_pretrained(root / "text_encoder")
    AutoencoderKLQwenImage21(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=64,
        dim_mult=[1, 1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True, True],
        latents_mean=[0.1] * 64,
        latents_std=[1.5] * 64,
    ).save_pretrained(root / "vae")
    FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True).save_pretrained(root / "scheduler")
    return root


def condition_model(snapshot, *, edit=False, offline=False):
    config = QwenImage21ConditionModelConfig(
        base_model_path=str(snapshot),
        height=32,
        width=32,
        num_train_timesteps=16,
        enable_edit=edit,
        seed=77,
        text_encoder_dtype="float32",
        vae_dtype="float32",
    )
    return QwenImage21ConditionModel(config, meta_init=offline).requires_grad_(False).eval()


def raw_batch(tmp_path, edit=False):
    Image.new("RGBA", (32, 32), (200, 40, 10, 64)).save(tmp_path / "target.png")
    record = {"prompt": "a red square", "image_path": "target.png"}
    if edit:
        Image.new("RGBA", (32, 32), (0, 80, 180, 128)).save(tmp_path / "reference.png")
        record["edit_images"] = ["reference.png"]
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(record) + "\n")
    dataset = build_dataset(
        "iterable",
        train_path=str(path),
        source_name="Qwen-Image-2.1-Edit" if edit else "Qwen-Image-2.1",
        transform=build_data_transform("qwen_image21_online", data_dir=str(tmp_path)),
        shuffle=False,
    )
    loader = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=1,
        global_batch_size=1,
        dataloader_batch_size=1,
        max_seq_len=512,
        train_steps=1,
        dyn_bsz=False,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        shuffle=False,
        collate_fn=DiTDataCollator(),
    )
    return next(iter(loader))[0]


def assert_tree_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_tree_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("edit", [False, True], ids=["text-to-image", "edit"])
def test_rgba_cache_offline_and_rng_restore(snapshot, tmp_path, edit):
    batch = raw_batch(tmp_path, edit)
    assert all(image.mode == "RGBA" for image in batch["images"][0])
    assert batch["images"][0][-1].getpixel((0, 0))[3] == 64
    if edit:
        assert batch["images"][0][0].getpixel((0, 0))[3] == 128
    online = condition_model(snapshot, edit=edit)
    start_rng = online.rng_state_dict()
    encoded = online.get_condition(**batch)
    assert_tree_equal(online.rng_state_dict(), start_rng)
    assert encoded["latents"][0].shape == (1, 128, 1, 2, 2)
    assert int(encoded["img_mask"][0].sum()) == int(edit)
    # Alpha reaches the VAE: changing only transparency must change its posterior.
    opaque_batch = {**batch, "images": [[image.copy() for image in batch["images"][0]]]}
    opaque_batch["images"][0][-1].putalpha(255)
    opaque = online.get_condition(**opaque_batch)
    assert not torch.equal(encoded["latents"][0], opaque["latents"][0])
    assert_tree_equal(online.rng_state_dict(), start_rng)

    # Exercise the real parquet serializer and offline transform, including nested reference tensors.
    cache = tmp_path / "cache"
    saver = OfflineEmbeddingSaver(str(cache), dataset_length=1)
    saver.save({key: value[0] for key, value in encoded.items()})
    row = Dataset.from_parquet(str(cache / "rank_0_shard_0.parquet"))[0]
    restored = DiTDataCollator()(build_data_transform("dit_offline")(row))
    assert_tree_equal(restored, encoded)
    offline = condition_model(snapshot, edit=edit, offline=True)
    assert offline.vae is None and offline.text_encoder is None and offline.processor is None
    expected = online.process_condition(**encoded)
    actual = offline.process_condition(**restored)
    assert_tree_equal(actual, expected)
    next_rng = online.rng_state_dict()
    next_batch = online.process_condition(**encoded)
    assert not torch.equal(next_batch["hidden_states"][0], expected["hidden_states"][0])
    offline.load_rng_state_dict(next_rng)
    assert_tree_equal(offline.process_condition(**restored), next_batch)
    posterior_mode = encoded["latents"][0][:, :64]
    normalized = ((posterior_mode - 0.1) / 1.5).flatten(2).transpose(1, 2)
    torch.testing.assert_close(actual["latents"][0], normalized)
    assert actual["training_target"][0].shape == normalized.shape
    # The same flow-matching identity holds with a clean reference prefix present.
    noisy_target = actual["hidden_states"][0][:, -4:]
    sigma = actual["timestep"][0].view(1, 1, 1)
    torch.testing.assert_close(noisy_target, normalized + sigma * actual["training_target"][0])
    if edit:
        reference = ((encoded["edit_latents"][0][0][:, :64] - 0.1) / 1.5).flatten(2).transpose(1, 2)
        torch.testing.assert_close(actual["hidden_states"][0][:, :4], reference)
    assert int(actual["img_mask"][0].sum()) * 4 == actual["hidden_states"][0].shape[1]


def test_offline_needs_only_component_metadata(snapshot, tmp_path):
    for component, filename in (("vae", "config.json"), ("scheduler", "scheduler_config.json")):
        (tmp_path / component).mkdir()
        shutil.copyfile(snapshot / component / filename, tmp_path / component / filename)
    offline = condition_model(tmp_path, offline=True)
    assert offline.text_encoder is None and offline.vae is None
    with pytest.raises(RuntimeError, match="offline_embedding"):
        offline.get_condition(inputs=["test"], images=[[Image.new("RGBA", (32, 32))]])


class _Runtime:
    def __init__(self, model, condition):
        self.model = model
        self.condition_model = condition
        self.parallel_state = parallel_state.get_parallel_state_by_name("base")

    def __call__(self, **kwargs):
        return self.model(**kwargs)


@pytest.mark.parametrize("edit", [False, True], ids=["text-to-image", "edit"])
@pytest.mark.parametrize("offline", [False, True], ids=["online", "offline"])
def test_native_trainer_updates_only_dit_and_roundtrips(snapshot, tmp_path, edit, offline):
    batch = raw_batch(tmp_path, edit)
    condition = condition_model(snapshot, edit=edit)
    if offline:
        batch = condition.get_condition(**batch)
        condition = condition_model(snapshot, edit=edit, offline=True)
    config = tiny_config()
    config.in_channels = config.out_channels = 64
    model = QwenImage21Transformer2DModel(config).train()
    model.gradient_checkpointing_enable()
    optimizer = build_optimizer(model, lr=1e-3)
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    trainer = DiTTrainer.__new__(DiTTrainer)
    trainer.base = SimpleNamespace(
        num_micro_batches=1,
        model=_Runtime(model, condition),
        device=torch.device("cpu"),
        LOG_SAMPLE=False,
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
    )
    trainer.training_task = "offline_training" if offline else "online_training"
    loss, losses = trainer.forward_backward_step(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    torch.testing.assert_close(loss, losses["mse_loss"])
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in condition.parameters())
    optimizer.step()
    assert any(not torch.equal(parameter, original[name]) for name, parameter in model.named_parameters())
    model.eval()
    inputs = condition.process_condition(**(batch if offline else condition.get_condition(**batch)))
    with torch.no_grad():
        expected = model(**inputs).predictions[0]
    checkpoint = tmp_path / "transformer"
    model.save_pretrained(checkpoint)
    reloaded = build_foundation_model(
        str(checkpoint),
        weights_path=str(checkpoint),
        init_device="cpu",
        torch_dtype="float32",
        ops_implementation=tiny_ops(),
    ).eval()
    with torch.no_grad():
        actual = reloaded(**inputs).predictions[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    get_device_type() not in ("cuda", "npu") or runner_device_count() < 2,
    reason="requires two accelerators for real FSDP2 gradient and update parity",
)
@pytest.mark.launch_plan(provider=CasePlan.sequential(_TWO_RANKS))
@pytest.mark.parametrize("mode", ["fp32", "bf16-gc"])
def test_accelerator_and_fsdp2_two_optimizer_steps(tmp_path, mode):
    environment = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            "-m",
            "tests.models.qwen_image21_fsdp_worker",
            str(tmp_path),
            mode,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "result.json").read_text())
    assert report == {
        "device": get_device_type(),
        "ranks": 2,
        "steps": 2,
        "mode": mode,
        "upstream_cpu": "covered-by-fp32-case" if mode == "bf16-gc" else "PASS",
        "fsdp2": "PASS",
    }


@pytest.mark.parametrize(
    ("resolution", "reference_size", "pixel_limits", "expected_size"),
    [
        (64, (64, 64), (65536, 16777216), (256, 256)),
        (64, (96, 32), (65536, 16777216), (448, 160)),
        (512, (96, 64), (65536, 16777216), (640, 416)),
        (1024, (96, 32), (1024, 65536), (416, 128)),
    ],
)
def test_reference_resize_matches_processor_grid(snapshot, resolution, reference_size, pixel_limits, expected_size):
    """VL and RGBA VAE must share the exact grid, including budget-rounding edges."""
    model = condition_model(snapshot, edit=True)
    model.config.height = model.config.width = resolution
    processor = model.processor.image_processor
    processor.size = {"shortest_edge": pixel_limits[0], "longest_edge": pixel_limits[1]}
    reference = Image.new("RGBA", reference_size, (100, 30, 20, 64))
    prepared = model._prepare_image(reference, reference=True)
    assert prepared.mode == "RGBA"
    assert prepared.size == expected_size
    assert prepared.getchannel("A").getextrema() == (64, 64)
    # The processor must leave this spatial grid unchanged when it sees the
    # white-composited RGB copy; otherwise image slots and VAE latents disagree.
    vision = processor(images=[prepared.convert("RGB")], return_tensors="pt")
    height, width = vision["image_grid_thw"][0, 1:].tolist()
    assert (width * processor.patch_size, height * processor.patch_size) == prepared.size
    parameters, shape = model._encode_image(prepared)
    assert shape == (1, prepared.height // 16, prepared.width // 16)
    assert parameters.shape[-2:] == shape[-2:]
    assert model._prepare_image(reference).size == (resolution, resolution)
