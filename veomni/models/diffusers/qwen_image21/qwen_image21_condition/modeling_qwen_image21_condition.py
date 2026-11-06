from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from veomni.data.multimodal.dit.data_transform import (
    resize_qwen_image21_reference_image,
    resize_qwen_image21_training_image,
)
from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.modeling_base import VeOmniPreTrainedModel
from veomni.utils.device import get_device_type

from .configuration_qwen_image21_condition import QwenImage21ConditionModelConfig


_TRAINING_TIME_SCALE = 1000
QWEN_IMAGE21_CONDITION_VERSION = 1
_MAX_TEXT_POSITIONS = 8192
_SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
_PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
_PROMPT_TEMPLATE = (
    f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{{references}}{{prompt}}{_PROMPT_TAIL}"
)


def _build_training_schedule(steps):
    """Build the DiffSynth Qwen-Image Flow Matching grid in FP32."""
    sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device="cpu")[:-1]
    shift = math.exp(0.8)
    sigmas = shift / (shift + (1 / sigmas - 1))
    one_minus_sigma = 1 - sigmas
    sigmas = 1 - one_minus_sigma / (one_minus_sigma[-1] / (1 - 0.02))
    timesteps = sigmas * _TRAINING_TIME_SCALE
    curve = torch.exp(-2 * ((timesteps - _TRAINING_TIME_SCALE / 2) / _TRAINING_TIME_SCALE) ** 2)
    curve = curve - curve.min()
    weights = curve * (_TRAINING_TIME_SCALE / curve.sum())
    if steps != _TRAINING_TIME_SCALE:
        weights = weights * (steps / _TRAINING_TIME_SCALE)
        weights = weights + weights[1]
    return sigmas, timesteps, weights


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
        seed = (config.seed or 0) + get_parallel_state().dp_rank
        self.generator = torch.Generator(device=get_device_type()).manual_seed(seed)
        self.timestep_generator = torch.Generator(device="cpu").manual_seed(seed)
        base = config.base_model_path
        self._sigmas, self._timesteps, self._loss_weights = _build_training_schedule(config.num_train_timesteps)
        self.vae_config = AutoencoderKLQwenImage21.load_config(base, subfolder=config.vae_subfolder)
        if self.vae_config.get("scale_factor_spatial", 16) != 16 or self.vae_config["z_dim"] != 64:
            raise ValueError("Qwen-Image 2.1 requires a 64-channel VAE with spatial compression factor 16.")
        # Offline training only needs the VAE normalization metadata.
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
        system_message = [{"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]}]
        system_tokens = self.processor.apply_chat_template(system_message, tokenize=True, return_dict=False)
        self._system_tokens = len(
            system_tokens if not system_tokens or isinstance(system_tokens[0], int) else system_tokens[0]
        )
        self._image_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]
        self._tail_tokens = len(self.processor.tokenizer.encode(_PROMPT_TAIL))

    @staticmethod
    def _as_list(value, length=None):
        if value is None:
            return [None] * length if length is not None else []
        return value if isinstance(value, list) else [value]

    def rng_state_dict(self):
        return {"generator": self.generator.get_state(), "timestep_generator": self.timestep_generator.get_state()}

    def load_rng_state_dict(self, state):
        self.generator.set_state(state["generator"].cpu())
        if "timestep_generator" in state:
            self.timestep_generator.set_state(state["timestep_generator"].cpu())

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

    def _prepare_image(self, image, *, target_size=None):
        if not isinstance(image, Image.Image):
            raise ValueError("Qwen-Image 2.1 expects PIL images loaded with alpha preserved.")
        image = resize_qwen_image21_training_image(
            image, height=self.config.height, width=self.config.width, max_pixels=self.config.max_pixels
        )
        if target_size is not None:
            size = self.processor.image_processor.size
            min_pixels = size.get("shortest_edge") if isinstance(size, dict) else getattr(size, "shortest_edge", None)
            image = resize_qwen_image21_reference_image(image, target_size=target_size, min_pixels=min_pixels)
        return image

    def _encode_prompt(self, prompt, references, reference_shapes):
        """Encode one interleaved prompt using the last decoder block's pre-norm states."""
        image_slots = " ".join(
            f"<image{i}><|vision_start|><|image_pad|><|vision_end|>" for i in range(1, len(references) + 1)
        )
        text = _PROMPT_TEMPLATE.format(references=image_slots, prompt=prompt or " ")
        processor_kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
        if references:
            rgb_references = []
            for image in references:
                white = Image.new("RGB", image.size, (255, 255, 255))
                white.paste(image, mask=image.getchannel("A"))
                rgb_references.append(white)
            processor_kwargs["images"] = rgb_references
        model_inputs = self.processor(**processor_kwargs)
        if references:
            vision_sizes = [math.prod(shape) for shape in model_inputs.image_grid_thw.tolist()]
            latent_sizes = [math.prod(shape) for shape in reference_shapes]
            if vision_sizes != latent_sizes:
                raise ValueError(
                    "Qwen3-VL and VAE reference token counts differ after resizing. "
                    "Adjust the reference aspect ratio or training resolution."
                )
        model_inputs = model_inputs.to(self.text_encoder.device)
        image_shift = sum(max(height, width) for _, height, width in reference_shapes)
        text_budget = _MAX_TEXT_POSITIONS - 1 - image_shift + self._system_tokens
        text_mask = model_inputs.attention_mask[0].bool() & model_inputs.input_ids[0].ne(self._image_token_id)
        overflow = int(text_mask.sum()) - text_budget
        if overflow > 0:
            # Mask prompt tokens before the encoder while keeping the assistant suffix and image slots.
            text_positions = text_mask.nonzero(as_tuple=True)[0]
            end = text_positions.numel() - self._tail_tokens
            model_inputs.attention_mask[0, text_positions[end - overflow : end]] = 0

        encoder_inputs = {"input_ids": model_inputs.input_ids, "attention_mask": model_inputs.attention_mask}
        if references:
            encoder_inputs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
        if "mm_token_type_ids" in model_inputs:
            encoder_inputs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        # The checkpoint consumes pre-norm features. Capture them directly from
        # Qwen3-VL's backbone and skip the language modeling head and KV cache.
        backbone = self.text_encoder.model
        hidden_states = None

        def capture_pre_norm(_module, args):
            nonlocal hidden_states
            hidden_states = args[0]

        with backbone.language_model.norm.register_forward_pre_hook(capture_pre_norm):
            backbone(**encoder_inputs, use_cache=False)
        positions = model_inputs.attention_mask[0].nonzero(as_tuple=True)[0][self._system_tokens :]
        embeddings = hidden_states.index_select(1, positions)
        image_mask = model_inputs.input_ids.index_select(1, positions).eq(self._image_token_id)
        return embeddings, image_mask

    def _encode_image(self, image):
        pixels = torch.from_numpy(np.array(image, dtype=np.float32)).to(device=self.vae.device, dtype=self.vae.dtype)
        # Scale in the VAE compute dtype, matching DiffSynth's preprocessing order.
        pixels = (pixels * (2 / 255) + (-1)).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        posterior = self.vae.encode(pixels).latent_dist
        return posterior.parameters, (1, posterior.parameters.shape[-2], posterior.parameters.shape[-1])

    @torch.no_grad()
    def get_condition(self, inputs, images, outputs=None, **kwargs) -> dict[str, Any]:
        if self.vae is None:
            raise RuntimeError("Offline training cannot encode raw inputs; run offline_embedding first.")
        condition = {
            key: []
            for key in (
                "condition_version",
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
            target = self._prepare_image(target)
            references = [self._prepare_image(image, target_size=target.size) for image in references]
            reference_params, reference_shapes = [], []
            for image in references:
                params, shape = self._encode_image(image)
                reference_params.append(params)
                reference_shapes.append(shape)
            embeddings, image_mask = self._encode_prompt(prompt, references, reference_shapes)
            if self.config.max_sequence_length is not None and embeddings.shape[1] > self.config.max_sequence_length:
                raise ValueError(
                    f"Encoded prompt length {embeddings.shape[1]} exceeds max_sequence_length="
                    f"{self.config.max_sequence_length}; increase the limit. Image slots cannot be truncated."
                )
            target_params, target_shape = self._encode_image(target)
            condition["condition_version"].append(QWEN_IMAGE21_CONDITION_VERSION)
            condition["latents"].append(target_params)
            condition["encoder_hidden_states"].append(embeddings)
            # Each encoder call removes padding from one sample.
            condition["encoder_hidden_states_mask"].append(None)
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
        latent = parameters[:, :64].to(self.generator.device)
        mean = torch.as_tensor(self.vae_config["latents_mean"], dtype=torch.float32, device="cpu").to(latent)
        std = torch.as_tensor(self.vae_config["latents_std"], dtype=torch.float32, device="cpu").to(latent)
        latent = (latent - mean.view(1, 64, 1, 1, 1)) / std.view(1, 64, 1, 1, 1)
        return latent.flatten(2).transpose(1, 2), (1, parameters.shape[-2], parameters.shape[-1])

    def _sample_training_inputs(self, target, target_shape):
        _, height, width = target_shape
        latent = target.transpose(1, 2).unflatten(2, (height, width))
        index = torch.randint(len(self._timesteps), (1,), device="cpu", generator=self.timestep_generator)
        timestep = self._timesteps[index].to(torch.bfloat16)
        # DiffSynth looks up sigma and the loss weight after rounding the raw time to BF16.
        nearest = (self._timesteps - timestep).abs().argmin()
        sigma = self._sigmas[nearest]
        noise = torch.empty_like(latent).normal_(generator=self.generator)
        noisy_latent = (1 - sigma) * latent + sigma * noise
        velocity = noise - latent
        return (
            noisy_latent.flatten(2).transpose(1, 2),
            velocity.flatten(2).transpose(1, 2),
            timestep.to(latent.device) / _TRAINING_TIME_SCALE,
            self._loss_weights[nearest],
        )

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
        condition_version=None,
        **kwargs,
    ):
        latents = self._as_list(latents)
        count = len(latents)
        if self._as_list(condition_version) != [QWEN_IMAGE21_CONDITION_VERSION] * count:
            raise ValueError("Incompatible Qwen-Image 2.1 condition cache. Regenerate it with offline_embedding.")
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
                "loss_weight",
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
            noisy_target, velocity, timestep, weight = self._sample_training_inputs(target, target_shape)
            result["loss_weight"].append(weight)
            image_mask = image_mask.to(device=target.device, dtype=torch.bool)
            result["hidden_states"].append(
                torch.cat([*packed_refs, noisy_target], dim=1) if packed_refs else noisy_target
            )
            result["training_target"].append(velocity)
            result["latents"].append(target)
            result["timestep"].append(timestep)
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
