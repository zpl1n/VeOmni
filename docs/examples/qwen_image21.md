# Qwen-Image 2.1 fine-tuning

This integration trains the base [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
7B diffusion transformer. Qwen3-VL and the RGBA VAE remain frozen. It supports
text-to-image and image-editing supervision, online encoding, and cached offline
training through `tasks/train_dit.py`. It is not a reproduction of an unpublished
Qwen training recipe, and does not implement reinforcement learning or Turbo distillation.

The native VeOmni DiT follows the
[DiffSynth-Studio Qwen-Image 2.1 architecture](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/models/qwen_image_21_dit.py).
Its single-stream blocks process the joint sequence and share one timestep
modulation projection. Frozen condition encoding calls Qwen3-VL and the RGBA
VAE directly, following the stages in the
[DiffSynth-Studio pipeline](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/qwen_image_21.py).

## Environment and weights

Use the project's locked environment: Diffusers 0.41.0 and Transformers 5.16.1.

```bash
uv sync --extra npu_aarch64 --dev  # Ascend ARM; use npu for x86, gpu for NVIDIA
source .venv/bin/activate
```

On Ascend, first source the `set_env.sh` from your installed CANN toolkit.
Download the complete base snapshot and retain its directory layout:

```text
Qwen-Image-2.1/
  transformer/   # config.json and both safetensors shards
  text_encoder/  # Qwen3-VL weights and config
  processor/     # tokenizer, image processor and chat template
  vae/           # 4-channel RGBA VAE, 64-channel latents
  model_index.json
```

`model.model_path` points to `transformer/`; `model.config_path` points to its
`config.json` or the matching [7B model config](../../configs/model_configs/qwen_image21/qwen_image21_7b.json)
in VeOmni. `model.condition_model_path` points to the snapshot root. All
components must come from the same snapshot. The transformer keeps the upstream
parameter names and saves an upstream-compatible Diffusers transformer config.

Model type names follow `qwen_image`: `QwenImage21Transformer2DModel` for the
transformer and `QwenImage21ConditionModel` for the condition model. Python package
names remain `qwen_image21_transformer` and `qwen_image21_condition`. When reusing
older configs or checkpoints, update their `model_type` and any explicit
`condition_model_type` fields to these class names.

## Data

Use JSONL records with a prompt and a target image. Relative image paths are
resolved under `data.mm_configs.data_dir`:

```json
{"prompt": "A red paper boat on transparent background", "target_image": "boat.png"}
```

For editing, provide one or more reference images and the desired target:

```json
{"prompt": "Change the boat to blue", "source_images": ["red_boat.png"], "target_image": "blue_boat.png"}
```

Select `data.source_name: Qwen-Image-2.1-Edit` and
`model.condition_model_cfg.enable_edit: true` for edit data. Apply both settings
in the embedding and offline stages too. Text-to-image uses `Qwen-Image-2.1`
and `enable_edit: false`. A run uses one of these two data layouts.

The `qwen_image21_online` transform uses its own RGBA loader to preserve alpha from files and encoded image
bytes. It feeds RGBA to the VAE. The prompt encoder composites reference images
over white for Qwen3-VL and reads the hidden states before final normalization.
Reference geometry also follows the image processor's
pixel limits and patch-grid rounding before either encoder runs. The released processor
requires at least 65,536 reference pixels (256 x 256 for a square), so small training
targets can have larger reference grids. The generic `dit_online` transform retains its RGB
default and should not be used for this model.

## Online fine-tuning

Start with the provided 512 x 512 example and adjust the paths:

```bash
MODEL=/path/to/Qwen-Image-2.1
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 \
  bash train.sh tasks/train_dit.py configs/dit/qwen_image21_sft.yaml \
  --model.model_path "$MODEL/transformer" \
  --model.config_path "$MODEL/transformer/config.json" \
  --model.condition_model_path "$MODEL" \
  --data.train_path /path/to/train.jsonl \
  --data.mm_configs.data_dir /path/to/images
```

The example uses FSDP2, BF16 computation with FP32 gradient reduction, gradient
checkpointing, micro batch size 1, and global batch size 8. Each rank processes
the complete sequence for its samples. Use the accelerator defaults
(`ulysses_size=1` and `cp_size=1`). Global batch size must be divisible by
`micro_batch_size * dp_size`. Reduce target resolution
before attempting larger batches. Both the frozen encoders and training state
consume accelerator memory in online mode.

## Embedding then offline training

Use `configs/dit/qwen_image21_embedding.yaml` with the same model and dataset
path overrides as above. This stage loads the frozen encoders without loading
the DiT or an optimizer. It writes Parquet shards under
`output/qwen_image21_embedding`; one epoch processes the dataset once. Set the
global batch size to match the chosen data-parallel launch size.

Then launch `configs/dit/qwen_image21_offline.yaml`, supplying the same three
model-path overrides and `--data.train_path output/qwen_image21_embedding`.
This stage reads the VAE configuration for latent normalization and samples
new noise and a timestep each step. Noise-generator state is included in the
framework's checkpoint extra state for resumption.

The cache stores raw VAE posterior parameters, pre-final-norm Qwen3-VL features,
attention masks and reference-image layouts. Existing Qwen-Image 2.1 Parquet
caches work with the native DiT. Regenerate the cache after changing the
images, prompts, resolution, crop policy, encoder weights or processor. Keep
`enable_edit` consistent across both stages. Parquet is a binary format; do not
edit the files as text. Only load caches you produced or trust: the existing
DiT cache format serializes tensor fields with pickle.

## Training semantics and limits

- This integration uses the VAE posterior mode for targets and references.
  Training uses the DiffSynth Flow Matching schedule directly, with uniform
  index sampling from a default 1,000 steps, fixed exponential shift `mu=0.8`,
  terminal sigma 0.02, and normalized timestep loss weights.
  The noisy target is `(1-sigma) * latent + sigma * noise`, and the target velocity
  is `noise - latent`. Loss is FP32 MSE over target tokens, weighted before averaging per sample.
- Reference latents stay clean and precede the noisy target. The official
  interleaving, causal text attention, and bidirectional attention within each
  image block are preserved. Reference and text positions receive no direct MSE.
- Height and width must be positive multiples of 32. Targets use the configured
  size and `image_resize_mode` (`center_crop` in the examples). References retain
  aspect ratio near the target area, subject to the processor pixel limits and
  patch-grid rounding described above. One latent token covers 16 x 16
  pixels: a 512 x 512 target contributes 1,024 tokens, in addition to prompt and
  reference-image tokens. VAE latent width is 64; DiT hidden width is 4,096.
- `max_sequence_length` in the condition config is an optional encoded-prompt
  limit. Exceeding it raises an error; it never truncates image slots. Generic
  `data.max_seq_len` does not resize images or truncate the DiT joint sequence.
- Supported attention selections are `eager` and `sdpa` (`veomni_sdpa` alias).
  Use SDPA for training; eager materializes attention scores. NPU rotary
  computation transfers real positional tables, avoiding complex NPU tensors.
- Tensor batches must share a layout. List samples are processed separately.

## Performance profiling

All three example YAMLs include a disabled `train.profile` section. Add
`--train.profile.enable true` to the training command to collect steps `[5, 7)`
(steps 5 and 6), with tensor shapes and memory statistics on rank 0. On Ascend,
the framework uses `torch_npu.profiler`; no model-code changes are required.
Traces are written under `output/qwen_image21_<stage>/profile`, where `<stage>`
is `sft`, `embedding`, or `offline`. Run enough batches to reach step 7 so the
collection can finish. Set `--train.profile.rank0_only false` for all-rank
communication analysis; this increases overhead and output size. Profiling adds
overhead, so measure final throughput separately with profiling disabled.

## Validation

The regression tests use locally generated tiny components, without downloading
models:

```bash
pytest -q tests/models/diffusers/test_qwen_image21.py \
  tests/models/test_qwen_image21_training.py
```

They compare upstream predictions and input/parameter gradients, strict checkpoint
loading and save/reload, RGBA preprocessing, real Qwen3-VL/VAE encoding, cache and
RNG restoration, and an optimizer step through the DiT trainer. The accelerator
cases require two devices and compare single-device execution with two-rank
FSDP2 over two optimizer steps in FP32 and in BF16 with gradient checkpointing.
These tests check numerical agreement on a tiny model.

The native 7.115B DiT matches all 297 tensor names and shapes in the released
checkpoint. The earlier Diffusers-backed implementation also ran a 64 x 64
full-model smoke check on two Ascend 950DT devices (torch/torch-npu 2.10.0,
Diffusers 0.41.0, Transformers 5.16.1), including real Qwen3-VL/VAE encoding
for generation and editing and one AdamW update with FSDP2, BF16 computation,
and gradient checkpointing. The native model's full-resolution throughput,
memory requirements, and image quality have yet to be measured.

If Ascend reports `EI0020` because its default HCCL port is occupied, set
`HCCL_NPU_SOCKET_PORT_RANGE=auto` in the launch environment, as described in the
[HCCL port documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_NPU_SOCKET_PORT_RANGE.md).
This was required for the validation host; it is separate from the model config.

The DiffSynth recipe follows `FlowMatchScheduler.set_timesteps_qwen_image` and
`set_training_weight` in [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/diffusion/flow_match.py).
Schedule construction stays on CPU FP32; sampled times and loss weights stay FP32
to avoid quantizing timestep lookup through BF16. This aligns the schedule and
loss equations, not bitwise RNG or all mixed-precision details of another framework.
Existing raw-posterior embedding caches can be reused. Remove the old
`training_recipe` field from Qwen-Image 2.1 configs. Runs started with the
earlier Diffusers schedule should resume on their original code revision to
preserve the training objective.
