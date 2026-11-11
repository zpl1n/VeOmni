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

"""Qwen-Image 2.1 MFU logging without changes to the shared DiT trainer."""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from veomni.trainer.callbacks import Callback
from veomni.trainer.dit_trainer import DiTTrainer, VeOmniDiTArguments
from veomni.utils import logging
from veomni.utils.count_flops import get_device_flops
from veomni.utils.device import synchronize
from veomni.utils.dist_utils import all_reduce

from .flops import QwenImage21FlopsCounter


logger = logging.get_logger(__name__)


@dataclass
class QwenImage21MFUConfig:
    peak_tflops: Optional[float] = field(
        default=None,
        metadata={
            "help": "Per-device dense peak TFLOP/s at the training precision; defaults to the BF16 device table."
        },
    )

    def __post_init__(self):
        if self.peak_tflops is not None:
            self.peak_tflops = float(self.peak_tflops)
            if not math.isfinite(self.peak_tflops) or self.peak_tflops <= 0:
                raise ValueError("mfu.peak_tflops must be finite and positive.")


@dataclass
class QwenImage21MFUArguments(VeOmniDiTArguments):
    mfu: QwenImage21MFUConfig = field(default_factory=QwenImage21MFUConfig)


class QwenImage21MFUCallback(Callback):
    """Wrap the existing meter, then publish DiT MFU before Tqdm and Wandb."""

    def __init__(self, trainer, meter_callback, peak_tflops):
        super().__init__(trainer)
        self.meter_callback = meter_callback
        self.module = trainer.model.unwrapped_module
        self.counter = QwenImage21FlopsCounter(trainer.model.model_config)
        self.peak_tflops = peak_tflops

    def on_step_begin(self, state, micro_batches=None, **kwargs):
        self.counter.reset()
        self.module._mfu_flops_counter = self.counter
        self.meter_callback.on_step_begin(state, micro_batches=micro_batches, **kwargs)
        synchronize()
        self.start_time = time.perf_counter()

    def on_step_end(self, state, **kwargs):
        synchronize()
        delta_time = time.perf_counter() - self.start_time
        self.module._mfu_flops_counter = None
        flops, tokens = all_reduce(
            (self.counter.total_flops, self.counter.total_tokens), op="sum", group=self.parallel_state.dp_group
        )
        delta_time = all_reduce(delta_time, op="max", group=self.parallel_state.fsdp_group)
        world_size = self.parallel_state.world_size
        achieved = flops / delta_time / 1e12

        self.meter_callback.on_step_end(state, **kwargs)
        env_metrics = self.trainer.step_env_metrics
        train_metrics = self.trainer.step_train_metrics
        env_metrics.update({"flops_achieved(T)": achieved, "mfu/step_time": delta_time, "mfu/joint_tokens": tokens})
        train_metrics["training/dit_tflops_per_device"] = achieved / world_size
        if self.peak_tflops is not None:
            promised = self.peak_tflops * world_size
            mfu = achieved / promised
            env_metrics.update({"flops_promised(T)": promised, "mfu": mfu})
            train_metrics["training/mfu(%)"] = 100 * mfu
            display_mfu = f"{100 * mfu:.3f}%"
        else:
            env_metrics.pop("mfu", None)
            env_metrics.pop("flops_promised(T)", None)
            display_mfu = "unavailable (set --mfu.peak_tflops)"
        env_metrics.update(train_metrics)
        if self.trainer.args.train.global_rank == 0:
            logger.info(
                f"Step {state.global_step}: DiT MFU={display_mfu}, "
                f"TFLOPS/device={achieved / world_size:.3f}, step_time={delta_time:.4f}s"
            )

    def on_train_end(self, state, **kwargs):
        self.module._mfu_flops_counter = None


class QwenImage21MFUTrainer(DiTTrainer):
    """Keep model-specific instrumentation outside the shared trainer."""

    def __init__(self, args: QwenImage21MFUArguments):
        if args.train.training_task == "offline_embedding":
            raise ValueError("MFU measures DiT training; use tasks/train_dit.py for offline_embedding.")
        super().__init__(args)
        if self.base.model.model_config.model_type != "QwenImage21Transformer2DModel":
            raise ValueError("This MFU entry point requires a Qwen-Image 2.1 transformer.")
        module = self.base.model.unwrapped_module
        if any(not parameter.requires_grad for parameter in module.parameters()):
            raise ValueError("The DiT MFU estimator requires full fine-tuning, without frozen DiT parameters or LoRA.")
        peak = args.mfu.peak_tflops
        mp = args.model.accelerator.fsdp_config.mixed_precision
        if args.model.accelerator.fsdp_config.fsdp_mode == "fsdp2" and mp.enable:
            compute_dtype = mp.param_dtype
        else:
            compute_dtype = str(module.dtype).removeprefix("torch.")
        if peak is None and compute_dtype == "bfloat16":
            detected = get_device_flops()
            if math.isfinite(detected) and detected > 0:
                peak = detected
        if args.train.global_rank == 0:
            logger.info(
                "DiT MFU counts analytical forward/backward matmuls; excludes frozen encoders and checkpoint "
                "recomputation. Step timing includes condition encoding, backward, communication and optimizer; "
                "excludes data fetching/broadcast and metric logging. "
                f"Compute dtype={compute_dtype}, dense peak/device={peak if peak is not None else 'unknown'} TFLOPS."
            )
        original = self.base.environ_meter_callback
        callback = QwenImage21MFUCallback(self.base, original, peak)
        self.base._callbacks[self.base._callbacks.index(original)] = callback
        self.base.environ_meter_callback = callback
