# LiteSpec release readiness review

Reviewed the current working tree on 2026-10-24. No production files were modified.

The tree is not ready for a release without addressing credential configuration leakage and license inconsistencies.

## P1: Credentials are submitted as experiment configuration

Locations: litespec/training/logging.py:225, litespec/training/logging.py:258, litespec/training/logging.py:290, litespec/training/logging.py:316, litespec/training/logging.py:397.

Mock W&B initialization confirmed both dummy wandb_key and swanlab_key appear in config. No real credentials or external services used.

Suggested action: Filter credential fields from a shared public configuration before passing it to any tracker.

## P1: License metadata and distributed license text disagree

Locations: pyproject.toml:11, LICENSE:1, litespec/modeling/target/custom_backend/qwen3.py:4, litespec/data/preprocessing.py:11.

Project metadata declares Apache-2.0; LICENSE contains MIT and sgl-project copyright. Apache-derived source exists; only MIT LICENSE is bundled in wheel.

Suggested action: Resolve project licensing intent and distribute applicable upstream license texts and attributions; preserve original notices and document modifications.

## P2: Source distribution omits files required by published documentation

Locations: MANIFEST.in:1, scripts/setup_environment.sh:16, README.md:58.

Built source archive contains setup_environment.sh but omits uv.lock and all assets, although the setup command uses uv sync --frozen and README references assets.

Suggested action: Include uv.lock and referenced documentation assets in the source distribution and smoke-test the extracted archive.

## Verification

- All 117 tracked Python files compile.
- CPU test run: 76 passed, 10 skipped, 3 warnings, 26.30 seconds. GPUs were hidden from the test process.
- An isolated build produced both a source archive and a wheel. Neither contains tests or benchmark evidence.
- Scanned 231 reachable Git blobs and current tracked text for selected credential patterns. Matches in embedded font data were false positives; no confirmed credential match. This is not an exhaustive secret audit.
- No tracked model weights, hidden caches, or dataset artifacts were found.
- New data collection assets remain untracked. No CI workflow exists.

## Coverage limits

No fresh full runtime installation, CUDA 12.8 regression, multi-GPU training, or exhaustive third-party provenance/font-license audit was performed. Prior GPU experiment receipts do not replace release regression for every configuration.
