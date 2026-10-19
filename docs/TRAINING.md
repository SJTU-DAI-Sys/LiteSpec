# LiteSpec training

## Package organization

- `litespec/modeling/draft/`: independent model classes and inference code;
shared FlexAttention, sampling, registry, and exportable dependency helpers.
RoPE stays in the algorithm implementations.
- `litespec/core/`: algorithm objectives plus shared block preparation, masks,
chunking, metrics, and `TrainingOutput` additive loss terms.
- `litespec/training/`: `config.py` handles CLI validation; `setup.py` prepares
data, target and FSDP; `runner.py` owns the complete training lifecycle.
`algorithms.py` defines the adapter and stage contracts. `recipes/` contains
algorithm-specific preparation and objectives, not independent training loops.
Optimizer/scheduler, logging/tracker, and checkpoint/export each have one module.
- `litespec/distributed/`: `groups.py` owns process groups, `collectives.py`
implements tensor operations, and `transport.py` owns packets and routing.
Model code continues to use the public `litespec.distributed` API unchanged.
- `litespec/args.py`: SGLang options shared with offline data generation.
- `scripts/`: thin algorithm entrypoints and six YAML-based shell launchers.

Configurations use exactly one algorithm key: `dflash_config`, `dflash2_config`,
`dspark_config`, or `dlite_config`. Model class names are unchanged. Only DLite
uses `mode: sft/teacher/two_stage`; other YAML files select `algorithm` only.

All six shell launchers accept `[config.yaml] [trainer arguments...]`. Later CLI
values override YAML, including `--no-profile`, `--no-disaggregate`, and other
BooleanOptionalAction flags. Final merged settings are validated before model
or distributed initialization. Topology comes from YAML or PET variables.

## DFlash-family training

`DFlashDraftModel`, `DFlash2DraftModel`, and `DSparkDraftModel` use independent
direct-training entrypoints while sharing the target capture, FSDP, offline
`regen_full`, checkpoint, and disaggregated execution infrastructure.

```bash
bash scripts/dflash/run_training_dflash.sh

# The other launchers are run_training_dflash2.sh and run_training_dspark.sh.
```

All three load their matching file under `scripts/config/` by default and
accept an optional YAML followed by CLI overrides. They accept `--init-from`,
`--resume-from`, and the common distributed options. Set `--disaggregate` together with the
target/draft rank topology to transport full target context hidden states
without transporting vocabulary logits. DFlash uses hard-label CE; DFlash2
adds the top-k selector objective; DSpark combines CE, distribution L1, and
confidence loss. DSpark uses anchor-inclusive blocks: output slot zero is not
supervised and slot one predicts the token after the anchor.

The shipped YAML recipes use five draft layers with block size 16 for DFlash
and block size 8 for DFlash2/DSpark. DFlash2 uses grouped convolution
`(kernel=2, group=16)` and selector `(rank=256, top-k=16)`. DSpark uses a
rank-256 vanilla Markov head with Markov-conditioned confidence.

Benchmarks use `evaluation/run_benchmark_dflash.sh`,
`evaluation/run_benchmark_dflash2.sh`, or `evaluation/run_benchmark_dspark.sh`.

## Common model settings

The YAML `arguments` section accepts these common settings; the equivalent CLI
option can be appended after the YAML to override it:


| YAML key                 | CLI option                 | Default                                     |
| ------------------------ | -------------------------- | ------------------------------------------- |
| `target_model_backend`   | `--target-model-backend`   | `hf`                                        |
| `attention_backend`      | `--attention-backend`      | `flex_attention`                            |
| `flex_attention_backend` | `--flex-attention-backend` | `TRITON`                                    |
| `block_size`             | `--block-size`             | `8`                                         |
| `num_draft_layers`       | `--num-draft-layers`       | `5`                                         |
| `swa_window_size`        | `--swa-window-size`        | teacher only                                |
| `chs_num_layers`         | `--chs-num-layers`         | `7`                                         |
| `target_layer_ids`       | `--target-layer-ids`       | automatically selected                      |
| `sequential_head`        | `--sequential-head`        | `rnn`                                       |
| `sequential_rank`        | `--sequential-rank`        | `256`                                       |
| `mask_token_id`          | `--mask-token-id`          | automatically selected for the target model |


CHS, SWA, and sequential-head options are DLite-specific; they are not arguments
of the other algorithms. SFT does not expose SWA, and two-stage training inherits
the teacher structure. DLite's mask is called the **block attention mask**; it is
separate from the sequential prediction head.

`sglang` remains available for target prefill and exposes the existing SGLang
memory, tensor-parallel, and draft-sharding options. The draft model uses
FlexAttention during training. `TRITON` uses natural Q/KV lengths; `FLASH`
pads Q/KV to full kernel tiles. Inference may use FlashAttention 2 or SDPA.

Set exactly one launcher data key. `train_data` tokenizes JSONL and runs target
prefill online. `train_hidden_states` reads a schema-v2
`regen_full` cache and loads only the target embedding and LM head. Offline
target logits are projected from cached final-norm hidden states at sampled
anchor positions. Offline mode requires `TP_SIZE=1` and
`shard_draft_by_tp: false`; torchrun workers then operate as data-parallel ranks.
Current caches also contain a compact per-sample loss-mask index. Startup
filters invalid samples from this JSON metadata without opening any sample
tensor member. Training then reads samples in shard-aware order, keeps worker
tar handles alive with a bounded LRU, pins collated CPU tensors, and performs
non-blocking GPU transfers. Schema-v2 caches created before the lightweight
index was added must be regenerated.

When `target_layer_ids` is non-empty, its unique, increasing, zero-based IDs  
are used directly and `chs_num_layers` is ignored. Set it to null to  
restore automatic evenly spaced selection by count. Qwen3.5 targets are loaded  
through SGLang with FA3 while their DLite draft remains a dense Qwen3-style  
backbone with matching vocabulary and hidden dimensions.

When `mask_token_id` is omitted, training preserves a value stored in a loaded
draft checkpoint, otherwise uses the tokenizer's native mask token, or selects
the first target embedding row not occupied by the tokenizer. An explicit
`MASK_TOKEN_ID` remains available as an override.

## Teacher

Run `scripts/dlite/run_training_dlite_teacher.sh`. Its loss can combine final-token CE,
L1 distribution distance, and base-LM CE using `FINAL_CE_WEIGHT`, `L1_LOSS_WEIGHT`, and
`BASE_LM_CE_WEIGHT`. CE labels use the target model's greedy prefill tokens,
while the `rnn` sequential head updates its state with ground-truth previous
tokens from the training sequence.

DLite uses `Q=[embed(anchor), MASK x (block_size-1)]` and a zero-initialized
sequential RNN state. `dlite_v1` assigns local position 0 to the anchor;
`dlite_v2` assigns position 1. CHS remains at position 0 in both versions, and
fresh or versionless configurations default to `dlite_v2`.

Teacher, Stage 1, transition, and Stage 2 training backpropagate additive loss
numerators. At each optimizer boundary, gradients are normalized by the sum of
effective loss weights across the complete gradient-accumulation window and all
ranks in the FSDP data-parallel process group. A final partial window uses its
actual accumulated denominator rather than treating missing microbatches as
zero-weight rank averages.

## Student

Run `scripts/dlite/run_training_dlite_two_stage.sh` and provide a trained teacher with
`TEACHER_DRAFT_PATH`.

- A new student backbone is always initialized from scratch.
- Stage 1 uses only `STAGE1_KL_WEIGHT`; it has no auxiliary regression or
true-label CE term.
- Stage 1 and Stage 2 share one dataloader built from the selected online JSONL
or offline `regen_full` cache.
- The teacher sequential head is copied into the student, then frozen for
Stage 1. The student backbone learns to match the teacher logits.
- Stage 2 unfreezes the student and uses `STAGE2_FINAL_CE_WEIGHT`,
`STAGE2_L1_WEIGHT`, and `STAGE2_BASE_CE_WEIGHT`.

Resume with `RESUME_FROM`. For fresh two-stage training,
`TEACHER_DRAFT_PATH`, `STAGE1_EPOCHS`, `STAGE2_EPOCHS`, and `LEARNING_RATE` are
required.

## Acceptance metrics

The only shared `train/` keys are `loss`, `learning_rate`,
`walk_accepted_length`, and `expected_accept_len`. Other values live under
`detail/<algorithm>/...` (including DLite stages). Per-position diagnostics keep
only `hard_label/unary_top16_recall`. Numerators and denominators are reduced
over the complete accumulation window and all training ranks.

Despite its retained public name, `train/walk_accepted_length` is a **block
average**, named `block_accepted_length` internally. Each supervised block counts
the anchor once plus consecutive correct predictions, stopping at the first
error or unsupervised position. Blocks without any supervised prediction are
excluded. Overlapping anchors are all counted: lengths `4, 1, 2` give `7/3`, not
a cross-anchor walk. The final selector/head prediction is used for each model.

All supervised DFlash-family and DLite stages record `expected_accept_len`.
For each predicted position, the acceptance probability is the distribution
overlap `sum_x min(p_target(x), p_draft(x))`; expected length is
`1 + sum_k prod_{i<=k} acceptance_i`, including the anchor. DSpark uses logits
after its Markov head, DFlash2 uses the normalized sparse selector proposal,
and DLite uses logits after its sequential head. Stage 1 of two-stage DLite is
teacher distillation only and therefore does not emit this target-vs-draft
metric.

## Checkpoints and resume

Exports contain the model's own configuration and complete inference-source
dependency closure; they do not require the `litespec` package to load through
Hugging Face remote code. Existing weights and outputs are not rewritten.

`training_state.pt` contains small shared metadata and scheduler information;
optimizer tensors, FP32 master parameters, and training RNG states remain in
per-rank files. Resume requires `litespec_training_v2` and all rank shards,
matching algorithm, DLite mode/stage, topology/TP/sharding, effective batch and
accumulation settings, optimizer parameter names/order/shapes, and scheduler
settings. Pre-refactor v1 training states are deliberately unsupported; existing
model exports can still initialize weights with `--init-from` where supported.
Old package/config names, common-only optimizer checkpoints,
scheduler migration, and partial optimizer restoration are unsupported.

DLite two-stage uses a continuous scheduler and a stable parameter order:
stage-1 backbone parameters followed by newly unfrozen stage-2 parameters.
Frozen parameters legitimately have no Adam state until their first update.
An existing output directory is preserved; a numbered sibling is reserved for
the new run.

The full loop is `training/runner.py:run_draft`, shared by ordinary batches and
disaggregated packets. One packet counts as one logical batch regardless of its
draft microbatch subdivision. Accumulation spans epochs, but flushes a tail at
each stage boundary. Logs aggregate the entire optimizer window.
Accuracy, acceptance length, distribution diagnostics, and loss-component
telemetry are computed only in windows that will be logged, according to
`log_interval`. Scheduled final logs and forced partial-window tail logs collect
metrics from every microbatch in that window. Ordinary non-log windows retain
only the training loss numerator/denominator; CE/L1/KL, selector objectives,
and DSpark confidence targets are still computed for training. Checkpoint
replay skips diagnostics. Direct model calls keep `collect_detailed_metrics=True`
by default; the shared runner passes the logging decision to every recipe. Periodic saves
retain the logical-step interval and occur only at optimizer boundaries; an
interval hit inside a pending window is not deferred. Stage-end cursors point to
the next stage, so resuming does not replay the completed transition.

This structural refactor preserves two existing DLite path differences: a
single-batch transition uses equal loss weights in ordinary training and only
the stage-1 weight in disaggregated training; teacher `max_steps` shortens the
disaggregated scheduler, but not the ordinary scheduler.

## Optional FSDP1 Muon + AdamW

All six training YAMLs expose `arguments.use_muon` (default `false`). Enable it
with `use_muon: true` or append `--use-muon` to an existing launch command. No
model/config architecture change or extra optimizer dependency is needed.

```yaml
use_muon: true
muon_lr: 0.02
muon_momentum: 0.95
muon_weight_decay: 0.1
muon_matrices_per_rank: 1
muon_adamw_lr: 0.0004
muon_adamw_weight_decay: 0.1
```

Newton–Schulz steps/coefficients, Nesterov, `eps`, the shape-LR rule, and
AdamW betas are fixed internals (`MUON_DEFAULTS`); they are not YAML/CLI options.

When enabled, `muon_lr` and `muon_adamw_lr` replace `learning_rate` for their
respective optimizers. Both follow the existing warmup/cosine schedule. When
disabled, all `muon_*` settings are ignored and the original AdamW behavior,
including its zero weight decay and `learning_rate`, is unchanged. CLI flags use
hyphens instead of underscores; CLI overrides appended to the launcher override
the YAML (including `--no-use-muon`).

Muon owns Attention/MLP/fc 2-D linear weights, convolution
`kernel_projection.weight`, and DLite teacher `history_fuse.weight`. AdamW owns
norms, biases, `base_kernel`, trainable embeddings, and **all** parameters under
selector, DSpark `markov_head`/`confidence_head`, and DLite `sequential_head`.
These module exceptions take precedence over tensor dimensionality. Norms and
biases use the configured AdamW weight decay too; there is no implicit no-decay
group. Unknown or conflicting ownership is an error.

After the accumulation window is normalized, one global norm clips both groups
together (`max_grad_norm`, default 1.0). Master weights and Muon momentum remain
FP32 local shards. Muon batches same-shape matrices, exchanges BF16 update
directions with two all-to-alls, orthogonalizes full matrices on assigned ranks,
and returns the updates to their original shards. Each matrix has its own norm
and Newton–Schulz iterations. LR scaling uses the **full** shape, while decoupled
weight decay uses the base LR. The math follows PyTorch 2.9.1's implementation.

`muon_matrices_per_rank: 1` already batches up to one matrix per rank in parallel;
larger values allow more matrices per compute rank and consume more temporary
memory. Up to two batches are in flight to overlap collection with computation.
The draft FSDP process group, not target ranks, participates in these exchanges.
The initial adapter supports the pinned PyTorch 2.9.1 FSDP1 configuration:
`use_orig_params=True`, `SHARD_GRAD_OP`, no CPU offload (single-rank `NO_SHARD`
is also accepted). Replicated trainable parameters outside FSDP are rejected.

`train/learning_rate` retains the AdamW LR; hybrid runs additionally log
`train/adamw_learning_rate` and `train/muon_learning_rate`. Frozen/no-gradient
parameters skip both optimizer state updates and weight decay. DLite two-stage
registration includes future-unfrozen parameters from the beginning.

Hybrid checkpoints retain rank-local master weights, both optimizers, both
schedulers, parameter ownership and shard layouts. Resume requires the same
topology and optimizer configuration. Switching between AdamW-only and hybrid
mode is **not** an exact resume and is rejected; start a new run using the
workflow's model initialization path instead. Existing AdamW-only checkpoints
remain supported with `use_muon: false`.

Validation/benchmark commands (run from the repository root):

```bash
PYTHONPATH=. .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 tests/distributed_muon_smoke.py
PYTHONPATH=. .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=3 tests/distributed_muon_smoke.py --subgroup
PYTHONPATH=. .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 tests/benchmark_fsdp_muon.py
```

The benchmark uses identical gradients and optimizer settings for serial-matrix
and batched routing, warms up and synchronizes GPUs, then reports per-step time,
peak allocated memory and matrix exchange calls. It is an optimizer-only
microbenchmark, not an end-to-end training speed claim.

## Disaggregated target/draft execution

Set `disaggregate: true` for online JSONL training in any algorithm YAML. The
shared settings are:


| YAML key                 | CLI option                 | Meaning                                    |
| ------------------------ | -------------------------- | ------------------------------------------ |
| `target_ranks_per_node`  | `--target-ranks-per-node`  | target ranks on each node                  |
| `draft_ranks_per_node`   | `--draft-ranks-per-node`   | draft FSDP ranks on each node              |
| `target_tp_size`         | `--target-tp-size`         | TP size of each independent target replica |
| `sglang_ep_size`         | `--sglang-ep-size`         | EP size inside each target replica         |
| `node_batch_size`        | `--node-batch-size`        | total samples produced per node and step   |
| `draft_micro_batch_size` | `--draft-micro-batch-size` | microbatch on each draft rank              |
| `pipeline_depth`         | `--pipeline-depth`         | in-flight packet slots, at least 2         |
| `profile`                | `--profile`                | collect bridge timing when true            |


`target_ranks_per_node` must be divisible by `target_tp_size`, and
`target_tp_size` must be divisible by `sglang_ep_size`. `node_batch_size` must
be divisible by both the target-replica count and `draft_ranks_per_node`.
TP/EP greater than one requires the SGLang backend. Pipeline depth is excluded
from checkpoint identity and may be changed on resume; topology, batch sizes,
model/layer configuration, and the training-data identity must match.

All process groups are created before ranks split into target and draft roles.
Target ranks retain the backend's sharded model head, but the training path does
not compute or transmit its vocabulary logits. Each draft rank instead loads a
full frozen target embedding and LM head, outside FSDP and the optimizer, and
projects the transmitted final-norm hidden states locally. Stage 1 packets omit
prediction logits entirely; transition, teacher, and SFT/Stage 2 compute them
only where their losses require them.

Disaggregated execution accepts `launcher.train_data` only. It is incompatible
with `launcher.train_hidden_states` and `shard_draft_by_tp: true`; `regen_full`
keeps using the ordinary offline DP/FSDP path.