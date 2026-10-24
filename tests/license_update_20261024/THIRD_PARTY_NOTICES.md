# Third-party notices

LiteSpec is distributed under the Apache License, Version 2.0; see [LICENSE](LICENSE).
Third-party portions retain their original copyright notices and license terms.
The project license does not replace those notices or claim ownership of upstream
code. Individual files can contain contributions from more than one source.

The entries below identify inherited and adapted implementation areas. A file
being listed does not mean that all of its contents were written by that upstream
project. Existing source-level copyright and attribution notices remain in place.

## SpecForge

- Source: <https://github.com/sgl-project/SpecForge>
- Copyright: Copyright (c) 2025 sgl-project.
- License: MIT; the inherited license is preserved without modification in
  [licenses/SpecForge-MIT.txt](licenses/SpecForge-MIT.txt).
- Relevant files include `litespec/data/preprocessing.py`,
  `litespec/data/utils.py`, `litespec/data/template.py`,
  `litespec/modeling/target/sglang_backend/model_runner.py`,
  `litespec/modeling/target/sglang_backend/patch.py`,
  `litespec/modeling/target/sglang_backend/utils.py`, and
  `litespec/layers/ring/ring_flash_attn.py`.
- These areas contain inherited framework code adapted for LiteSpec data
  preparation, target execution, and distributed execution. Pre-existing
  third-party notices within inherited files continue to apply, including the
  Apache-licensed portions described below.

## SGLang

- Source: <https://github.com/sgl-project/sglang>
- License and upstream notice:
  [licenses/SGLang-Apache-2.0.txt](licenses/SGLang-Apache-2.0.txt).
- `litespec/data/template.py` identifies SGLang's chat-template implementation
  as a source.
- `litespec/modeling/target/sglang_backend/utils.py` identifies the SGLang
  v0.5.4 logits processor as the basis of its modified hidden-state output.
- `litespec/modeling/target/sglang_backend/model_runner.py` and `patch.py`
  adapt runner initialization and distributed-group integration for LiteSpec.
- LiteSpec modifications integrate template registration, hidden-state capture,
  and execution with LiteSpec interfaces and process groups.

## Hugging Face Transformers and GPT-NeoX

- Sources: <https://github.com/huggingface/transformers> and
  <https://github.com/EleutherAI/gpt-neox>.
- License: Apache-2.0. The Transformers license and upstream notice are in
  [licenses/Transformers-Apache-2.0.txt](licenses/Transformers-Apache-2.0.txt);
  the complete Apache-2.0 terms are also in the root [LICENSE](LICENSE).
- The following target implementations retain their original source-level
  notices: `litespec/modeling/target/custom_backend/gpt_oss.py`, `llama.py`,
  `llama4.py`, `phi3.py`, `qwen2.py`, `qwen3.py`, and `qwen3_moe.py` in the
  same directory.
- `litespec/data/preprocessing.py` and `litespec/data/utils.py` retain their
  EleutherAI and Hugging Face notices and identify upstream processing code.
- `litespec/modeling/draft/flex_attention.py` references the Transformers
  flexible-attention implementation.
- LiteSpec modifications adapt target execution, hidden-state access,
  attention integration, and data preparation for this framework. The original
  notices identifying Hugging Face, EleutherAI, Microsoft, Qwen, and other
  contributors remain in the applicable files.

## EAGLE

- Source: <https://github.com/SafeAILab/EAGLE>
- Copyright: Copyright 2025 SafeAI Lab (SAIL).
- License and upstream notice:
  [licenses/EAGLE-Apache-2.0.txt](licenses/EAGLE-Apache-2.0.txt).
- `litespec/data/preprocessing.py` and `litespec/data/utils.py` identify EAGLE
  as a source of adapted data-processing code. LiteSpec integrates these
  routines with its training-data preparation.

## verl

- Source: <https://github.com/volcengine/verl>
- License: [licenses/Verl-Apache-2.0.txt](licenses/Verl-Apache-2.0.txt).
- `litespec/distributed/collectives.py` identifies the gather implementation
  in `verl/utils/ulysses.py` at commit
  `a0e8e4472b8b472409defb0c8fcc5162301450af` as a source.
- LiteSpec modifications integrate these operations with LiteSpec process groups.

## YunChang and ring-flash-attention

- Sources: <https://github.com/feifeibear/long-context-attention> and
  <https://github.com/zhuzilin/ring-flash-attention>.
- Licenses: [YunChang Apache-2.0](licenses/YunChang-Apache-2.0.txt) and
  [ring-flash-attention MIT](licenses/Ring-Flash-Attention-MIT.txt).
- `litespec/layers/ring/__init__.py` identifies YunChang as an upstream source.
  `ring_flash_attn.py` and `utils.py` in that directory contain the adapted
  ring-attention implementation; `utils.py` also references the upstream
  ring-flash-attention discussion of output and log-sum-exp accumulation.
- LiteSpec modifications integrate ring attention with the framework's attention
  interfaces. The ring-flash-attention notice preserves Copyright 2024 Zilin Zhu.

## Installed dependencies and model artifacts

Installed dependencies retain their own licenses; this file documents the
third-party implementation areas identified in the source tree. Model weights
and datasets are not included in the LiteSpec package and have their own terms.
