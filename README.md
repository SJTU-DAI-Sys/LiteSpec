![LiteSpec logo](assets/logo.svg)

# LiteSpec

LiteSpec is a unified training framework for four independent speculative
decoding algorithms: **DFlash, DFlash2, DSpark, and DLite**. They share training,
data, distributed execution, metrics, and checkpoint infrastructure while
retaining their own model configurations, architectures, and inference logic.
The installation and Python package name is `litespec`. See
[the training guide](docs/TRAINING.md) for online, offline, and disaggregated examples.

DLite supports SFT, teacher training, two-stage student training, and standalone
evaluation. The student predicts a block in
parallel and uses a lightweight `rnn` sequential head to restore token-to-token
dependence inside the block.

The repository contains the complete training stack. The
[DLite](https://github.com/SJTU-DAI-Sys/DLite) repository is the smaller
inference-only release.

```bash
uv sync --frozen --extra cu130 --no-dev
source .venv/bin/activate
```



## Environment

The validated environment is Linux x86_64 with:

- CPython 3.11 (the repository pins it in `.python-version`);
- an NVIDIA GPU with BF16 support and a driver compatible with the selected CUDA runtime (12.8 or 13.0);
- [uv](https://docs.astral.sh/uv/) 0.11 or newer.

The `cu130` and `cu128` extras connect to different dependency configurations.
Select exactly one; do not combine packages from the two environments.

```bash
# A fresh CUDA 13 environment (default):
uv sync --frozen --extra cu130 --no-dev
source .venv/bin/activate
# Or install the CUDA 12.8 route to a separate environment:
UV_PROJECT_ENVIRONMENT=.venv-cu128 uv sync --frozen --extra cu128 --no-dev
source .venv-cu128/bin/activate
```

FlashAttention is optional. Install the regular dependencies first so PyTorch
is available while FlashAttention is built:

```bash
uv pip install flash-attn --no-build-isolation
```

The supported target backends are `hf` and `sglang`. Attention backend options
provided by PyTorch/Transformers are preserved; FlashAttention falls back to
SDPA when it is unavailable.

## Train

All training entrypoints use their matching YAML by default:

```bash
bash scripts/dflash/run_training_dflash.sh
bash scripts/dflash2/run_training_dflash2.sh
bash scripts/dspark/run_training_dspark.sh
bash scripts/dlite/run_training_dlite_sft.sh
bash scripts/dlite/run_training_dlite_teacher.sh
bash scripts/dlite/run_training_dlite_two_stage.sh
```

Edit the corresponding file under `scripts/config/`, or provide another YAML.
Training CLI arguments after the optional YAML override its `arguments` values:

```bash
bash scripts/dlite/run_training_dlite_teacher.sh \
  scripts/config/dlite_teacher.yaml \
  --batch-size 2 --tp-size 2 --shard-draft-by-tp
```

Both stages consume the same `TRAIN_DATA_PATH`. The student backbone always
starts from random initialization. Stage 1 trains only with weighted KL
distillation; Stage 2 uses the configured CE/TV objectives. The teacher's
sequential head is copied to the student and frozen during Stage 1.

All six launchers also support offline `regen_full` training. In YAML, replace
`launcher.train_data` with `launcher.train_hidden_states`. The two data keys are mutually exclusive. 

Online training can instead disaggregate target prefill from draft training.
Configure the shared `disaggregate` block in YAML. For an 8-GPU TP2/EP2
topology, use six target ranks, two draft ranks, and `sglang_ep_size: 2`.
It is online-only and cannot be combined with `shard_draft_by_tp: true`.

```bash
bash scripts/dlite/run_training_dlite_teacher.sh --disaggregate
```

See [docs/TRAINING.md](docs/TRAINING.md) for configuration details.

## Training Results

With a Qwen3-8B target model, LiteSpec achieves higher iteration throughput than SpecForge baseline in both online and offline training. With on-policy sampling and the integration of Muon optimizer, LiteSpec also achieves better training accuracy than baseline:

![LiteSpec and SpecForge training accuracy comparison](assets/training_acc_comparison.svg)![Qwen3-8B target training throughput comparison](assets/training-throughput-comparison.svg)

## Data collection

LiteSpec overlaps data preparation, model computation, and storage to collect
hidden states efficiently. For 4,000 samples, optimization reduces collection
time from 1,420.4 to 325.7 seconds for Qwen3-8B (**4.36× faster**) and from
2,253.4 to 448.2 seconds for Qwen3.8-27B (**5.03× faster**).

![End-to-end hidden-state collection time for 4,000 samples before and after optimization](assets/data-collection-comparison.svg)

The pipeline prepares requests using their actual lengths, computes only valid tokens, and overlaps model execution with asynchronous copies and background writes. Direct tensor storage avoids expensive per-sample serialization and final repacking.

![Reading, model forward, and writing overlap across batches in the asynchronous collection pipeline](assets/data-collection-pipeline.svg)

See [the optimization notes](docs/OPTIMIZATION.md)
for the main techniques. Times are end-to-end measurements from one run per
version, including initialization and completion of pending writes.

## Benchmark

```bash
TARGET_MODEL=/path/to/Qwen3-8B \
DRAFT_NAME_OR_PATH=/path/to/student/final \
DATASET=gsm8k MAX_SAMPLES=100 \
bash evaluation/run_benchmark_dlite.sh
```

Checkpoint model settings live under `dlite_config`. The public architecture is
`DLiteDraftModel`; `dlite_v1` uses local anchor position 0 and `dlite_v2` uses
local anchor position 1 while CHS remains at position 0.

## Roadmap

- [x] Support the Muon optimizer
- [ ] Support more target models (Qwen3.8, Gemma, ...)
- [ ] Support more draft architectures and let them be mixed freely
  - Attention: MLA, DSA, KDA, ...
  - Modules: convolution, selector, sequential head, ...



## Acknowledgements

We thank **[SpecForge](https://github.com/sgl-project/SpecForge)** for providing an excellent framework for rapidly validating new ideas and for its foundational training implementation. We also thank **[Hugging Face Transformers](https://github.com/huggingface/transformers)** for the open-source model ecosystem that supports LiteSpec's model workflows.

## Citation

```bibtex
@article{xu2026dlite,
  title={DLite: History-free Speculative Decoding for Agentic AI},
  author={Xu, Jiaming and Wang, Hanzhen and Zhao, Tianlang and Xiao, Jiewen and Yuan, Chengze and Pan, Jiayi and Zhou, Yongkang and Ye, Jiancai and Wang, Yu and Dai, Guohao},
  journal={arXiv preprint arXiv:2610.23787},
  year={2026}
}
```

