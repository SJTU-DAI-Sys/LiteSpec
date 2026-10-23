<p align="center">
  <img src="assets/logo.svg" alt="LiteSpec logo" width="80%">
</p>

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
bash scripts/setup_environment.sh cu130
source .venv-cu130/bin/activate
```



## Environment

The validated environment is Linux x86_64 with:

- CPython 3.11 (the repository pins it in `.python-version`);
- an NVIDIA GPU with BF16 support and a driver compatible with the selected CUDA runtime (12.8 or 13.0);
- a CUDA 12.x or 13.x toolkit matching the selected PyTorch CUDA major version when extensions must be built from source;
- [uv](https://docs.astral.sh/uv/) 0.11 or newer.

The CUDA 13 route pins PyTorch 2.11.0, TorchVision 0.26.0, TorchAudio
2.11.0, Transformers 5.12.1 and SGLang 0.5.16. SGLang's matching kernel
and FlashInfer dependencies are resolved by `uv.lock`; extensions are rebuilt
against the selected PyTorch ABI.

The CUDA 12.8 route pins PyTorch 2.9.1, TorchVision 0.24.1, TorchAudio
2.9.1, Transformers 5.3.0 and SGLang 0.5.10. A single version combination
cannot cover both routes: SGLang 0.5.16 requires CUDA Python 13, while
PyTorch 2.11's cu128 wheel requires CUDA bindings below 13. Select exactly
one extra; do not combine packages from the two environments.

```bash
# A fresh CUDA 13 environment (default):
bash scripts/setup_environment.sh cu130
source .venv-cu130/bin/activate
# Or install the CUDA 12.8 route to a separate environment:
bash scripts/setup_environment.sh cu128 /path/to/env-cu128
```

The setup script runs `uv sync --frozen` and `uv pip check`. Set `CUDA_HOME`
to a toolkit matching the chosen CUDA major version before building extensions.
On hosts using NVIDIA forward compatibility, add the installed compatible
`libcuda` directory to `LD_LIBRARY_PATH` when launching processes; the script
does not replace host drivers or shared environments. Verify imports, attention,
collectives and forward/backward on the deployment hardware before benchmarking.

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
`launcher.train_data` with `launcher.train_hidden_states`, and set `tp_size: 1`
plus `shard_draft_by_tp: false`. The two data keys are mutually exclusive.
Offline mode skips target-transformer prefill and computes
target logits from cached final-norm hidden states with the frozen LM head.
Newly generated caches include a lightweight per-sample supervision index, so
startup filtering does not deserialize token or hidden-state tensors. Offline
loading is shard-aware and uses persistent workers, pinned batches, and
non-blocking device copies. Regenerate older caches that lack this index.

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

<p align="center">
  <img src="assets/training_acc_comparison.svg" alt="LiteSpec and SpecForge training accuracy comparison" width="49%">
  <img src="assets/training-throughput-comparison.svg" alt="Qwen3-8B target training throughput comparison" width="49%">
</p>

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

We thank [**SpecForge**](https://github.com/sgl-project/SpecForge) for providing an excellent framework for rapidly validating new ideas and for its foundational training implementation. We also thank [**Hugging Face Transformers**](https://github.com/huggingface/transformers) for the open-source model ecosystem that supports LiteSpec's model workflows.

## License

See [LICENSE](LICENSE).

## Citation
```bibtex
@article{xu2026dlite,
  title={DLite: History-free Speculative Decoding for Agentic AI},
  author={Xu, Jiaming and Wang, Hanzhen and Zhao, Tianlang and Xiao, Jiewen and Yuan, Chengze and Pan, Jiayi and Zhou, Yongkang and Ye, Jiancai and Wang, Yu and Dai, Guohao},
  journal={arXiv preprint arXiv:2610.23787},
  year={2026}
}
```


## Hidden capture pipeline

`prepare_hidden_states.py` keeps the default synchronous `tar-v2` path.
For SGLang capture and overlapped persistence, use:

```bash
torchrun --standalone --nproc-per-node=2 scripts/data/prepare_hidden_states.py \
  --target-model-path /models/target --target-model-backend sglang \
  --data-path /data/responses.jsonl --output-path /local/cache \
  --tp-size 2 --batch-size 4 --target-layer-ids 1,16,31,46,61,63 \
  --sglang-attention-backend fa3 \
  --capture-mode async --cache-format tar-raw-v1 \
  --target-shard-bytes 2147483648 --queue-depth 2 --writer-workers 2 \
  --pending-bytes 8589934592 --profiling-output /local/capture_timings.json
```

`fa3` selects SGLang's FlashAttention 3 implementation; installing the separate
`flash-attn` Python package is not required for that backend. Hybrid models keep
their native linear-attention implementation alongside FA3 full-attention layers.

`--capture-request-layout auto` sends true lengths for ordinary attention models.
For hybrid linear-attention models it preserves padded request lengths, then
packs only valid output tokens. This keeps the dense reference calculation:
removing padding failed the strict per-layer numerical gate for Qwen3.8-27B.
`--capture-request-layout packed` explicitly opts into true-length requests and
must pass model-specific numerical checks before use. Padding preservation does
not change the saved sequence lengths, dtype, tokens or masks.

All TP ranks forward together; the TP owner alone copies and writes full-width
hidden states. `tar-raw-v1` preserves BF16 values in versioned tensor byte slabs
inside ordinary tar members. The manifest maps samples to member offsets across
owner shards; training ranks need not match capture ranks. The offline reader
supports both legacy tar and raw caches and exposes the same training fields.
`--token-budget` and `--length-bucketing` are optional; zero budget and original
order retain fixed batch sizing. `--resume` validates committed shards and only
recaptures uncommitted samples. The final manifest is published after pending
copies and durable commits complete. Checksums are calculated while writing;
normal completion does not scan all hidden files again.

Stage timings overlap and must not be added together. The benchmark end-to-end
clock includes process startup, model load, preprocessing and final persistence.
Download and archival clocks are reported separately.


RoPE frequencies remain FP32 when draft weights are cast to BF16 and when FSDP
casts buffers. This follows Transformers 5's loading behavior, so a fresh draft
and its checkpoint reload use the same positional frequencies. Model weights and
captured hidden tensors remain BF16.
