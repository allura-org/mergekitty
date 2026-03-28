# Changelog

Historical entries before `0.3.2rc2` are backfilled from git history. Where several patch or release-candidate bumps landed without a changelog update, they are grouped into a single release-series entry.

## 0.3.2rc2 - 2026-03-28

### Changed

- Replaced the old GPU boolean flags with explicit `--compute-device` and `--storage-device` options for shared merge CLIs.
- Added `--load-to-compute` so merges can load tensors directly onto the compute device while still storing completed tensors on the configured storage device.
- Updated merge planning and executor result handling so load, compute, and storage device choices are applied consistently across YAML and LoRA merge flows.
- Updated related scripts and README usage examples to match the new device option model.

### Tests

- Added CLI regression coverage for the new device arguments, invalid choices, and rejection of the removed legacy flags.
- Added planner and LoRA regression tests covering storage-device loads, compute-device loads via `--load-to-compute`, and command reconstruction for the new flags.

## 0.3.2rc1 - 2026-03-28

### Fixed

- Fixed an architecture bug in model matching and added regression coverage for the failure case.

## 0.3.1 Series - 2026-03-22

### Added

- Added architecture definitions for Gemma 3, Qwen 3.5 Dense, and Apertus models.

### Changed

- Allowed tied embeddings for Apertus models in follow-up patch releases.

## 0.3.0 - 2026-03-22

### Changed

- Modernized the dependency floor across the project, including newer Torch, Transformers, Accelerate, Tokenizers, Gradio, Ruff, and pre-commit releases.

## 0.2.3 - 2026-03-22

### Fixed

- Rolled back from later `0.2.2` candidates to the `0.2.2rc2` performance profile to avoid a memory leak and speed regressions.

### Changed

- Landed follow-up documentation and packaging cleanup after the rollback.

## 0.2.2 Series - 2026-03-22

### Changed

- Added shard-aware tensor load ordering, per-worker lazy tensor reader state, an async shard writer, and configurable writer queue depth for executor-driven merges.
- Tuned executor scheduling defaults and related graph execution behavior for better throughput.

### Tests

- Added regression coverage for graph ordering, lazy tensor loading, and tensor writer behavior introduced by the executor pipeline work.

## 0.2.1 - 2026-03-22

### Added

- Added executor-based LoRA merging via `mergekitty-merge-lora`.
- Added Qwen2-VL support with templated pre/post weights for multimodal merges.
- Added architecture support for OLMoE, GLM4, Mistral3/Ministral3, and broader MoE families including Mixtral, Qwen3, and Qwen3-MoE.
- Added an initial Gradio GUI prototype.

### Changed

- Introduced a parallel executor and reworked tensor loading and writing internals for multi-worker execution.

### Tests

- Added dedicated architecture, MoE merge, graph, lazy loader, I/O, and LoRA merge regression coverage for the new executor and architecture work.

## 0.0.7 - 2025-02-07

### Changed

- Renamed packages, imports, scripts, and project metadata from `mergekit` to `mergekitty`.
- Switched builds to `hatch` and formatting/linting to `ruff`.
- Made `tokenizer_source` default to `"base"` and moved merges onto the tokenizer-source workflow.
- Refactored executor code into the `mergekitty.executor` package.

### Removed

- Removed fallback tokenizer copying.
- Removed `bakllama`, `mergekit-legacy`, and `mergekit-evolve`.
- Folded `nuslerp` into `slerp`, removing the older SLERP implementation while keeping support for both `t` and `weight` style parameters.

# FORKED HERE! THE BELOW IS BACKFILLED FROM THE ORIGINAL MERGEKIT COMMITS! :3

## 0.0.6 - 2025-01-24

### Added

- Added NuSLERP, SCE merging, NearSwap, and additional merge methods from the `2405.07813` paper.
- Added `--load-in-4bit` and `--load-in-8bit` flags for the Hugging Face evaluation backend.
- Added support for preserving `modules_to_save` in `mergekit-extract-lora`.
- Added a decorator to simplify custom merge method definitions.

### Changed

- Improved optional-weight and tied-weight handling, including padding embeddings to configurable multiples.

### Fixed

- Fixed NuSLERP tokenizer merge behavior in a follow-up patch release.

## 0.0.5 Series - 2024-10-29 to 2024-11-30

### Added

- Added Cloud Merging plus support for InternLM2, Solar, and Exaone architectures.
- Added activation-based merging, the Della merge method, and improved `extract_lora.py` workflows.
- Added support for Phi-3 Small and updated Llama architecture handling for 1B and 3B variants.

### Changed

- Overhauled tokenizer merging, added explicit output chat template control, and improved list-based merge metadata handling.
- Improved tied-weight, optional-weight, and Cohere LM head handling.

## 0.0.4 Series - 2024-01-23 to 2024-06-28

### Added

- Added the overhauled computation graph, `mergekit-moe`, `mergekit-tokensurgeon`, `mergekit-extract-lora`, and `mergekit-evolve`.
- Added JSON architecture definitions and support for Baichuan, Falcon, Qwen2, Gemma, StarCoder2, HF Mamba, StableLM, Cohere, BERT, DistilBERT, RoBERTa, Phi-3, and Gemma2 families.
- Added Model Stock, Model Breadcrumbs, generic sparsification rescaling, LoRA extraction, MoE merge rework, and the `uniform_random` MoE gate mode.
- Added `out_dtype` support for controlling output tensor dtype.

### Changed

- Improved MoE merge behavior, upload metadata, and architecture packaging as the toolset expanded beyond basic text-only merges.

## 0.0.3 Series - 2023-11-25 to 2024-01-22

### Added

- Added ChatGLM, StableLMEpoch, Phi-2, and GPT-2 sequence-classification architecture support.
- Added DARE merging with reproducible `random_seed`, lazy PyTorch unpickling, generated Hugging Face model cards, `mergekit-mega`, and safer serialization controls.
- Added tokenizer merge tests, basic merge tests, and a PyTest GitHub Action.

### Changed

- Moved CLI argument parsing from Typer to Click and improved tokenizer configuration around `tokenizer_source`.

### Fixed

- Fixed CPU lazy-unpickle and tokenizer merge issues, plus multiple `trust_remote_code` propagation regressions.

## 0.0.2 Series - 2023-10-05 to 2023-10-24

### Added

- Added packaged CLI releases with output sharding, verbose logging, GPT-2 support, task arithmetic, and the `mergekit-layershuffle` utility.
- Added `--write-yaml`, `--full-random`, and related workflow options for generating or scrambling merge configs.

### Changed

- Added stronger architecture checks and output config handling as the CLI surface expanded.

## 0.0.1 - 2023-10-05

### Added

- Initial packaged release of the project, including linear merge and SLERP-based merge methods.
- Added per-model density controls, layer gradients, tokenizer addition and copying, notebook examples, and Kaggle-oriented cache handling.

### Fixed

- Included early fixes for LoRA/base-model handling and configuration edge cases before the first published package series stabilized.
