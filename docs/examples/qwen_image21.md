# Qwen-Image 2.1 fine-tuning

This integration trains the base [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
7B diffusion transformer. Qwen3-VL and the RGBA VAE remain frozen. It supports
text-to-image and image-editing supervision, online encoding, and cached offline
training through `tasks/train_dit.py`. It is not a reproduction of an unpublished
Qwen training recipe, and does not implement reinforcement learning or Turbo distillation.

The native VeOmni DiT follows the
[DiffSynth-Studio Qwen-Image 2.1 architecture](https://github.com/modelscope/DiffSynth-Studio/blob/08fdc0da8f41426885d9432e9007e6e460a4c30d/diffsynth/models/qwen_image_21_dit.py).
Its single-stream blocks process the joint sequence and share one timestep
modulation projection. Frozen condition encoding calls Qwen3-VL and the RGBA
VAE directly, following the stages in the
[DiffSynth-Studio pipeline](https://github.com/modelscope/DiffSynth-Studio/blob/08fdc0da8f41426885d9432e9007e6e460a4c30d/diffsynth/pipelines/qwen_image_21.py).

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

Use `qwen_image21_online` to preserve alpha when loading files or encoded image
bytes. The condition model resizes and center-crops each target and reference
using the DiffSynth training dataset geometry. It then resizes the prepared
references with Lanczos to approximately the target area, rounding each dimension
to a multiple of 32. The released processor requires at least 65,536 reference
pixels (256 x 256 for a square), so small targets can have larger reference grids.
If the processor changes the reference token count after resizing, adjust the
reference aspect ratio or target resolution so the Qwen3-VL slots match the VAE.
The VAE receives RGBA, and Qwen3-VL receives references composited over white.
The prompt encoder masks excess text before the forward pass, preserves the
image slots and assistant suffix, and reads the last layer before final normalization.

## Online fine-tuning

The default uses the DiffSynth full-training example's dynamic resolution with
a 1,048,576-pixel cap. For a fixed 512 x 512 crop, set both `height` and `width`
under `model.condition_model_cfg` in the YAML.
Set `data.mm_configs.data_dir` to your image directory in the YAML. For online
training, adjust the remaining paths in this command.

```bash
MODEL=/path/to/Qwen-Image-2.1
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 \
  bash train.sh tasks/train_dit.py configs/dit/qwen_image21_sft.yaml \
  --model.model_path "$MODEL/transformer" \
  --model.config_path "$MODEL/transformer/config.json" \
  --model.condition_model_path "$MODEL" \
  --data.train_path /path/to/train.jsonl
```

The example uses FSDP2, BF16 computation with FP32 gradient reduction, gradient
checkpointing, micro batch size 1, and global batch size 8. Both the frozen
encoders and training state consume accelerator memory in online mode.

## Sequence parallelism

The training examples default to complete sequences on each rank. For long
sequences, Qwen-Image 2.1 supports synchronous Ulysses SP. On eight devices,
add these overrides to the online or offline training command:

```bash
  --model.accelerator.ulysses_size 4 \
  --train.global_batch_size 2
```

As in Qwen-Image, the existing `ulysses_size` selects sequence-parallel execution.
Its default of 1 uses the complete sequence on each rank. Keep `cp_size` at its
default of 1; context parallelism is unsupported for this model.

The command processes two samples per micro step, each shared by four
sequence-parallel ranks. Global batch size must be divisible by
`micro_batch_size * dp_size`, where `dp_size = world_size / ulysses_size`.
Keep the global batch size at 8 to accumulate four micro batches per rank in
this example. The number of attention heads must be divisible by
`ulysses_size`.

The model slices the sequence before the Transformer blocks. As in Qwen-Image,
each Q/K/V tensor uses `[batch, sequence, heads, head_dim]` for Ulysses all-to-all,
then applies full-sequence RoPE. It reverses the exchange after attention and
gathers the hidden states, removing padding before the final normalization and
output projection. The target predictions feed the weighted loss. Qwen-Image 2.1
retains the DiffSynth single interleaved image/text stream and block-causal masks.
Each online sequence-parallel peer runs the frozen condition encoders
for its group's sample. Embedding can be run with the original topology
because the cached conditions store full samples.
Keep the same sequence-parallel group size when resuming a training checkpoint
so each group continues to share a noise and timestep stream. When changing
the group size, start a new run from the exported model weights.

## Embedding then offline training

Use `configs/dit/qwen_image21_embedding.yaml` with the same model and dataset
path overrides as above. This stage loads the frozen encoders without loading
the DiT or an optimizer. It writes Parquet shards under
`output/qwen_image21_embedding`; one epoch processes the dataset once. Set the
global batch size to match the chosen data-parallel launch size.

Then launch `configs/dit/qwen_image21_offline.yaml`, supplying the same three
model-path overrides and `--data.train_path output/qwen_image21_embedding`.
This stage reads the VAE configuration for latent normalization and samples
new noise and a timestep each step. The framework saves both generators in the
checkpoint extra state for resumption.

The cache stores raw VAE posterior parameters, pre-final-norm Qwen3-VL features,
attention masks and reference-image layouts. Regenerate caches produced before
the DiffSynth preprocessing and BF16 alignment. The offline trainer checks the
cache version and reports when a new embedding run is needed. Use a fresh
output directory for the new shards and point offline training to that directory.
Regenerate the cache after changing the images, prompts, resolution, encoder
weights, dtype or processor. Keep `enable_edit` consistent across both stages.
Only load caches you produced or trust because the format serializes tensors with pickle.

## Training semantics and limits

- This integration uses the VAE posterior mode for targets and references.
  Training uses the DiffSynth Flow Matching schedule directly, with uniform
  index sampling from a default 1,000 steps, fixed exponential shift `mu=0.8`,
  terminal sigma 0.02, and normalized timestep loss weights. It rounds the
  sampled raw timestep to BF16, uses the nearest FP32 schedule point for sigma
  and loss weight, and divides the rounded time by 1,000 for the DiT.
  It samples noise in BCHW layout and computes `(1-sigma) * latent + sigma * noise`
  and the velocity `noise - latent` before packing. The loss applies the sampled
  weight to FP32 MSE on the BCHW target prediction, then averages per sample.
- Reference latents stay clean and precede the noisy target. The official
  interleaving, causal text attention, and bidirectional attention within each
  image block are preserved. Reference and text positions receive no direct MSE.
- Fixed height and width must be positive multiples of 32. With both unset,
  each image keeps its aspect ratio up to resizing to fit `max_pixels` and
  rounding down to multiples of 32. Use fixed dimensions when an image would
  round down to a zero-width or zero-height crop. One latent token covers 16 x 16
  pixels: a 512 x 512 target contributes 1,024 tokens, in addition to prompt and
  reference-image tokens. VAE latent width is 64; DiT hidden width is 4,096.
- `max_sequence_length` in the condition config is an optional encoded-prompt
  limit. Exceeding it raises an error; it never truncates image slots. Generic
  `data.max_seq_len` does not resize images or truncate the DiT joint sequence.
- Supported attention selections are `eager` and `sdpa` (`veomni_sdpa` alias).
  The training YAMLs use `veomni_sdpa`, which selects the same PyTorch SDPA
  computation. Eager materializes attention scores. NPU rotary uses real
  positional tables, avoiding complex NPU tensors.
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
cases are configured to compare single-device execution with two-device FSDP2
over two optimizer steps in FP32 and in BF16 with gradient checkpointing, using
plain FSDP2 and Ulysses SP. The four-process CPU case compares Ulysses SP
(`ulysses_size=2`, `dp_size=2`) against the unsharded model, with and without
gradient checkpointing, including padding, weighted loss, parameter gradients,
and updates. These Ulysses-only cases need a fresh run after removing CP support.

An independent CPU comparison against the pinned DiffSynth implementation
matched the BF16 condition features, clean and noisy latents, sampled time and
weight, prediction, weighted loss, and parameter gradients exactly on tiny
text-to-image and two-reference edit cases. Both cases used the same SDPA
backend, model weights, timestep index, and BCHW noise. The released BF16 VAE
also matched DiffSynth on 32 x 32 and 64 x 96 RGBA inputs.

The native 7.115B DiT matches all 297 tensor names and shapes in the released
checkpoint. The earlier Diffusers-backed implementation also ran a 64 x 64
full-model smoke check on two Ascend 950DT devices (torch/torch-npu 2.10.0,
Diffusers 0.41.0, Transformers 5.16.1), including real Qwen3-VL/VAE encoding
for generation and editing and one AdamW update with FSDP2, BF16 computation,
and gradient checkpointing. Full-resolution throughput, memory requirements, and
image quality have yet to be measured.

If Ascend reports `EI0020` because its default HCCL port is occupied, set
`HCCL_NPU_SOCKET_PORT_RANGE=auto` in the launch environment, as described in the
[HCCL port documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_NPU_SOCKET_PORT_RANGE.md).
This was required for the validation host; it is separate from the model config.

For cross-framework comparisons, use the same weights, prepared sample, sampled
index, BCHW noise, dtype, and device backend. Set
`DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch` on the DiffSynth side to match VeOmni's
SDPA selection. VeOmni uses per-rank generators and FSDP2, so its seed identifies
a VeOmni run. Resume runs started with the previous preprocessing or timestep
rules on their original revision to preserve their training inputs.
