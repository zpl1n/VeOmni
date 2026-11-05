"""Training schedules and noise sampling for Qwen-Image 2.1."""

import math

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


class QwenImage21FlowMatch:
    def __init__(self, config):
        self.steps = config.num_train_timesteps
        self._schedule_key = None
        self._scheduler = None
        if config.training_recipe == "diffusers":
            self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                config.base_model_path, subfolder=config.scheduler_subfolder
            )
            self.time_scale = self._scheduler.config.num_train_timesteps
        else:
            self.time_scale = 1000

    def _diffsynth_schedule(self):
        # DiffSynth's Qwen-Image training grid has a fixed exponential shift
        # and terminal sigma. Build in FP32 before moving to the training device.
        sigmas = torch.linspace(1.0, 0.0, self.steps + 1, dtype=torch.float32, device="cpu")[:-1]
        shift = math.exp(0.8)
        sigmas = shift / (shift + (1 / sigmas - 1))
        one_minus_sigma = 1 - sigmas
        sigmas = 1 - one_minus_sigma / (one_minus_sigma[-1] / (1 - 0.02))
        timesteps = sigmas * self.time_scale
        curve = torch.exp(-2 * ((timesteps - 500) / 1000) ** 2)
        curve = curve - curve.min()
        weights = curve * (1000 / curve.sum())
        if self.steps != 1000:
            weights = weights * (self.steps / 1000)
            weights = weights + weights[1]
        return sigmas, timesteps, weights

    def _diffusers_schedule(self, image_seq_len, device):
        cfg = self._scheduler.config
        base_len, max_len = cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096)
        if base_len == max_len:
            raise ValueError("Scheduler base_image_seq_len and max_image_seq_len must differ.")
        base_shift, max_shift = cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15)
        mu = base_shift + (image_seq_len - base_len) * (max_shift - base_shift) / (max_len - base_len)
        self._scheduler.set_timesteps(self.steps, device=device, mu=mu)
        return self._scheduler.sigmas, self._scheduler.timesteps, None

    def prepare_schedule(self, image_seq_len, device):
        # Only the legacy Diffusers schedule depends on the target grid.
        key = (image_seq_len if self._scheduler is not None else None, device)
        if key != self._schedule_key:
            if self._scheduler is None:
                sigmas, timesteps, weights = self._diffsynth_schedule()
            else:
                sigmas, timesteps, weights = self._diffusers_schedule(image_seq_len, device)
            self.sigmas = sigmas.to(device)
            self.timesteps = timesteps.to(device)
            self.loss_weights = None if weights is None else weights.to(device)
            self._schedule_key = key

    def sample(self, target, generator):
        """Return the noisy input, velocity target, normalized time and loss weight."""
        self.prepare_schedule(target.shape[1], target.device)
        noise = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
        indices = torch.randint(len(self.timesteps), (1,), device=target.device, generator=generator)
        sigma = self.sigmas[indices].to(target.dtype).view(1, 1, 1)
        noisy_target = (1 - sigma) * target + sigma * noise
        timestep = self.timesteps[indices] / self.time_scale
        weight = None if self.loss_weights is None else self.loss_weights[indices]
        return noisy_target, noise - target, timestep, weight
