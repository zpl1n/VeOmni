from __future__ import annotations

import math
from typing import Any

import torch
from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from PIL import Image, ImageOps
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.modeling_base import VeOmniPreTrainedModel
from veomni.utils.device import get_device_type

from .configuration_qwen_image21_condition import QwenImage21ConditionModelConfig


class QwenImage21ConditionModel(VeOmniPreTrainedModel):
    """Frozen Qwen3-VL/RGBA VAE encoding and flow-matching input construction."""

    config_class = QwenImage21ConditionModelConfig
    supports_gradient_checkpointing = False

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        return attn_implementation

    def __init__(self, config: QwenImage21ConditionModelConfig, meta_init=False, **kwargs):
        super().__init__(config, **kwargs)
        self.processor = None
        self.text_encoder = None
        self.vae = None
        self.pipeline = None
        self.generator = torch.Generator(device=get_device_type())
        self.generator.manual_seed((config.seed or 0) + get_parallel_state().dp_rank)
        self._timesteps_image_seq_len = None
        base = config.base_model_path
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base, subfolder=config.scheduler_subfolder)
        self.vae_config = AutoencoderKLQwenImage21.load_config(base, subfolder=config.vae_subfolder)
        if self.vae_config.get("scale_factor_spatial", 16) != 16 or self.vae_config["z_dim"] != 64:
            raise ValueError("Qwen-Image 2.1 requires a 64-channel VAE with spatial compression factor 16.")
        # Offline training loads only scheduler/normalization metadata, never the encoders' weights.
        if meta_init:
            return
        self.processor = Qwen3VLProcessor.from_pretrained(base, subfolder=config.processor_subfolder)
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            base,
            subfolder=config.text_encoder_subfolder,
            dtype=getattr(torch, config.text_encoder_dtype),
            attn_implementation=config.text_encoder_attn_implementation,
        ).eval()
        self.vae = AutoencoderKLQwenImage21.from_pretrained(
            base, subfolder=config.vae_subfolder, torch_dtype=getattr(torch, config.vae_dtype)
        ).eval()
        self.requires_grad_(False)
        # Reuse the checkpoint's exact prompt, pre-final-norm features and white-alpha
        # compositing. The inference denoising loop (__call__) is never invoked.
        self.pipeline = QwenImage21Pipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            processor=self.processor,
            transformer=None,
        )

    @staticmethod
    def _as_list(value, length=None):
        if value is None:
            return [None] * length if length is not None else []
        return value if isinstance(value, list) else [value]

    def rng_state_dict(self):
        return {"generator": self.generator.get_state()}

    def load_rng_state_dict(self, state):
        self.generator.set_state(state["generator"].cpu())

    def _split_image_inputs(self, inputs, outputs, images):
        samples = self._as_list(inputs)
        if not samples or len(samples) != len(images):
            raise ValueError("Qwen-Image 2.1 needs matching nonempty batches of inputs and images.")
        if outputs is not None and len(outputs) != len(samples):
            raise ValueError("Input and output sample counts differ.")
        result = []
        for index, (entries, sample_images) in enumerate(zip(samples, images)):
            if isinstance(entries, str) and not self.config.enable_edit:
                entries, sample_outputs = [("text", entries)], [("image", None)]
            else:
                sample_outputs = outputs[index] if outputs is not None else None
            if not isinstance(entries, (list, tuple)) or not isinstance(sample_outputs, (list, tuple)):
                raise ValueError("Inputs/outputs must contain (modality, value) pairs.")
            texts, count = [], 0
            for entry in entries:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError("Inputs must contain (modality, value) pairs.")
                modality, value = entry
                if modality == "text" and isinstance(value, str):
                    texts.append(value)
                elif modality == "image" and value is None:
                    count += 1
                else:
                    raise ValueError("Qwen-Image 2.1 supports text and image placeholders only.")
            if (
                len(texts) != 1
                or len(sample_outputs) != 1
                or not isinstance(sample_outputs[0], (list, tuple))
                or tuple(sample_outputs[0]) != ("image", None)
            ):
                raise ValueError("Each sample requires one prompt and one output image.")
            if self.config.enable_edit != (count > 0):
                raise ValueError("enable_edit must match the presence of reference images in the sample.")
            sample_images = self._as_list(sample_images)
            if len(sample_images) != count + 1:
                raise ValueError("Images must follow reference placeholders, then the target placeholder.")
            result.append((texts[0], sample_images[:-1], sample_images[-1]))
        return result

    def _prepare_image(self, image, *, reference=False):
        if not isinstance(image, Image.Image):
            raise ValueError("Qwen-Image 2.1 expects PIL images loaded with alpha preserved.")
        image = image.convert("RGBA")
        if reference:
            width, height, _ = calculate_dimensions(self.config.height * self.config.width, image.width / image.height)
            if min(width, height) <= 0:
                raise ValueError("Reference aspect ratio is too extreme for the configured image resolution.")
            processor = self.processor.image_processor
            if processor.do_resize:
                # The VLM may enlarge small references or shrink large ones. Feed
                # that same grid to the VAE, including smart_resize's ceil/floor
                # behavior at non-square pixel-budget boundaries.
                height, width = smart_resize(
                    height,
                    width,
                    factor=processor.patch_size * processor.merge_size,
                    min_pixels=processor.size["shortest_edge"],
                    max_pixels=processor.size["longest_edge"],
                )
        else:
            width, height = self.config.width, self.config.height
            if self.config.image_resize_mode == "center_crop":
                return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        return self.pipeline.image_processor.resize(image, height=height, width=width)

    def _encode_image(self, image):
        pixels = self.pipeline.image_processor.preprocess(image, height=image.height, width=image.width).unsqueeze(2)
        posterior = self.vae.encode(pixels.to(device=self.vae.device, dtype=self.vae.dtype)).latent_dist
        return posterior.parameters, (1, posterior.parameters.shape[-2], posterior.parameters.shape[-1])

    @torch.no_grad()
    def get_condition(self, inputs, images, outputs=None, **kwargs) -> dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Offline training cannot encode raw inputs; run offline_embedding first.")
        condition = {
            key: []
            for key in (
                "latents",
                "encoder_hidden_states",
                "encoder_hidden_states_mask",
                "img_mask",
                "img_shapes",
                "edit_latents",
                "edit_img_shapes",
            )
        }
        # Per-sample encoding handles different reference layouts without mixing masks.
        for prompt, references, target in self._split_image_inputs(inputs, outputs, images):
            references = [self._prepare_image(image, reference=True) for image in references]
            target = self._prepare_image(target)
            embeddings, mask, image_mask = self.pipeline.encode_prompt(
                prompt=prompt,
                image=references or None,
                device=self.text_encoder.device,
            )
            if self.config.max_sequence_length is not None and embeddings.shape[1] > self.config.max_sequence_length:
                raise ValueError(
                    f"Encoded prompt length {embeddings.shape[1]} exceeds max_sequence_length="
                    f"{self.config.max_sequence_length}; increase the limit. Image slots cannot be truncated."
                )
            target_params, target_shape = self._encode_image(target)
            reference_params, reference_shapes = [], []
            for image in references:
                params, shape = self._encode_image(image)
                reference_params.append(params)
                reference_shapes.append(shape)
            condition["latents"].append(target_params)
            condition["encoder_hidden_states"].append(embeddings)
            condition["encoder_hidden_states_mask"].append(mask)
            condition["img_mask"].append(image_mask)
            condition["img_shapes"].append([target_shape])
            condition["edit_latents"].append(reference_params)
            condition["edit_img_shapes"].append(reference_shapes)
        return condition

    def _pack_posterior(self, parameters):
        if parameters.ndim != 5 or parameters.shape[:3] != (1, 128, 1):
            raise ValueError("Qwen-Image 2.1 cache requires raw VAE posterior parameters [1, 128, 1, H, W].")
        if any(size % 2 for size in parameters.shape[-2:]):
            raise ValueError("Latent height and width must be even to match Qwen3-VL image slots.")
        # The official reference-image encoding uses the posterior mode. Use the
        # same deterministic convention for targets; randomness belongs to FM noise/t.
        latent = DiagonalGaussianDistribution(parameters).mode().to(self.generator.device)
        mean = torch.as_tensor(self.vae_config["latents_mean"], device=latent.device, dtype=latent.dtype)
        std = torch.as_tensor(self.vae_config["latents_std"], device=latent.device, dtype=latent.dtype)
        latent = (latent - mean.view(1, 64, 1, 1, 1)) / std.view(1, 64, 1, 1, 1)
        return latent.flatten(2).transpose(1, 2), (1, parameters.shape[-2], parameters.shape[-1])

    def _prepare_timesteps(self, image_seq_len):
        if self._timesteps_image_seq_len == image_seq_len:
            return
        cfg = self.scheduler.config
        base_len, max_len = cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096)
        if base_len == max_len:
            raise ValueError("Scheduler base_image_seq_len and max_image_seq_len must differ.")
        base_shift, max_shift = cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15)
        mu = base_shift + (image_seq_len - base_len) * (max_shift - base_shift) / (max_len - base_len)
        self.scheduler.set_timesteps(self.config.num_train_timesteps, device=self.generator.device, mu=mu)
        self._timesteps_image_seq_len = image_seq_len

    @torch.no_grad()
    def process_condition(
        self,
        latents,
        encoder_hidden_states,
        img_shapes,
        img_mask,
        encoder_hidden_states_mask=None,
        edit_latents=None,
        edit_img_shapes=None,
        **kwargs,
    ):
        latents = self._as_list(latents)
        count = len(latents)
        fields = [
            self._as_list(encoder_hidden_states),
            self._as_list(img_shapes),
            self._as_list(img_mask),
            self._as_list(encoder_hidden_states_mask, count),
            self._as_list(edit_latents, count),
            self._as_list(edit_img_shapes, count),
        ]
        if not count or any(len(field) != count for field in fields):
            raise ValueError("Every Qwen-Image 2.1 cache field must have the same nonzero sample count.")
        result = {
            key: []
            for key in (
                "hidden_states",
                "timestep",
                "encoder_hidden_states",
                "encoder_hidden_states_mask",
                "img_shapes",
                "img_mask",
                "training_target",
                "latents",
            )
        }
        for parameters, context, shapes, image_mask, context_mask, refs, ref_shapes in zip(latents, *fields):
            target, target_shape = self._pack_posterior(parameters)
            if [tuple(shape) for shape in shapes] != [target_shape]:
                raise ValueError("Cached target img_shapes does not match its VAE posterior grid.")
            if context.ndim != 3 or context.shape[0] != 1 or image_mask.shape != context.shape[:2]:
                raise ValueError("Cached prompt embeddings/mask must have shape [1, S, D] / [1, S].")
            if context_mask is not None and context_mask.shape != context.shape[:2]:
                raise ValueError("Cached encoder_hidden_states_mask must match the prompt length.")
            refs, ref_shapes = refs or [], ref_shapes or []
            if len(refs) != len(ref_shapes) or self.config.enable_edit != bool(refs):
                raise ValueError("Cached reference images must match edit_img_shapes and enable_edit.")
            packed_refs = []
            for params, shape in zip(refs, ref_shapes):
                packed, actual_shape = self._pack_posterior(params)
                if tuple(shape) != actual_shape:
                    raise ValueError("Cached reference grid does not match its VAE posterior.")
                packed_refs.append(packed)
            if int(image_mask.bool().sum()) * 4 != sum(ref.shape[1] for ref in packed_refs):
                raise ValueError("Qwen3-VL reference image slots do not match the VAE reference latent count.")
            self._prepare_timesteps(target.shape[1])
            noise = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=self.generator)
            indices = torch.randint(
                len(self.scheduler.timesteps),
                (1,),
                device=target.device,
                generator=self.generator,
            )
            timestep = self.scheduler.timesteps[indices]
            sigma = self.scheduler.sigmas[indices].to(target.dtype).view(1, 1, 1)
            noisy_target = (1 - sigma) * target + sigma * noise
            image_mask = image_mask.to(device=target.device, dtype=torch.bool)
            result["hidden_states"].append(torch.cat([*packed_refs, noisy_target], dim=1))
            result["training_target"].append(noise - target)
            result["latents"].append(target)
            result["timestep"].append(timestep / self.scheduler.config.num_train_timesteps)
            result["encoder_hidden_states"].append(context.to(target.device))
            result["encoder_hidden_states_mask"].append(
                None if context_mask is None else context_mask.to(target.device)
            )
            result["img_shapes"].append([*[tuple(shape) for shape in ref_shapes], target_shape])
            result["img_mask"].append(
                torch.cat(
                    [
                        image_mask,
                        image_mask.new_ones(1, math.prod(target_shape) // 4),
                    ],
                    dim=1,
                )
            )
        return result
