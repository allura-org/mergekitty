```
# mergekitty

`mergekitty` is a toolkit for merging pre-trained language models. It uses an out-of-core approach so you can run surprisingly complex merges on modest hardware — entirely on CPU, or with as little as 8 GB of VRAM.

## What's this fork?

Forked from [mergekit](https://github.com/arcee-ai/mergekit) (originally by Charles Goddard, then maintained by Arcee.ai). The original project switched to a BSL license after a ton of community contribution, then switched back to LGPL but added a CLA that lets them relicense at will. So here we are.

## What changed?

A few things from upstream mergekit:

- All names/imports/scripts renamed to `mergekitty` (find-replace and you're good)
- VLM support with templated pre/post-weights (architecture files are incompatible with mergekit's)
- `tokenizer_source` now defaults to `"base"`; legacy tokenizer copying is gone
- `nuslerp` → `slerp` (old slerp removed). Supports both `t` (SLERP) and `weight` (NuSLERP) params
- `bakllama`, `mergekit-legacy`, and `mergekit-evolve` removed
- LoRA merging script via `mergekitty-merge-lora`
- Switched to `ruff` for formatting/linting and `hatch` for builds

## Why merge models?

Model merging is chaos magick. Done right, the result is better than any of its inputs. It's been proven repeatedly and nobody fully understands why. Ship it.

## Features

- Works with Llama 3, Qwen 3 (Dense & MoE), Mistral, GLM4, GPT-NeoX, BERT, and more
- [Tons of merge methods](#merge-methods) — arguably too many
- GPU or CPU — your call
- Lazy tensor loading for low memory use
- Interpolated gradient parameters for fine control
- Layer-stacking / "Frankenmerging" (à la Goliath, Midnight Miqu)
- [MoE merging](#mixture-of-experts-merging) and [LoRA extraction](#lora-extraction)

## Install

```sh
# recommended — isolated tool install
uv tool install mergekitty

# or just pip
pip install mergekitty

# from source
git clone https://github.com/allura-org/mergekitty.git
cd mergekitty
pip install -e .
```

## Usage

```sh
mergekitty-yaml path/to/config.yml ./output-model [--cuda] [--lazy-unpickle] [--allow-crimes]
```

Run `mergekitty-yaml --help` for the full list of options.

### Sharing on Huggingface

mergekitty generates a `README.md` for your merge. Edit it, keep it as-is, whatever — then upload:

```sh
huggingface-cli login
huggingface-cli upload your_username/my-cool-model ./output-model .
```

## Merge Configuration

Configs are YAML. The main fields:

| Field | Description |
|---|---|
| `merge_method` | Which algorithm to use ([see below](#merge-methods)) |
| `slices` / `models` | Input model definitions (mutually exclusive) |
| `base_model` | Base model, for methods that need one |
| `parameters` | Weights, densities, etc. — specifiable at multiple levels |
| `dtype` | Data type for the merge |
| `tokenizer` | Vocabulary and embedding configuration |
| `chat_template` | Override the output chat template |

### Parameters

Parameters (`weight`, `density`, etc.) can be set at four levels, most-specific wins:

1. `slices.*.sources.parameters` — per input slice
2. `slices.*.parameters` — per output slice
3. `models.*.parameters` — per input model
4. `parameters` — global fallback

Values can be scalars or interpolated gradients (a list of floats for smooth transitions across layers).

### Tokenizer

Use the `tokenizer` field for full control, or `tokenizer_source` for the simple legacy behavior.

```yaml
tokenizer:
  source: union          # "union", "base", or a model path
  tokens:                # optional: per-token embedding overrides
    :
      source: "chatml_model"
    <|start_header_id|>:
      source: "llama3_model"
      force: true
  pad_to_multiple_of: null
```

Defaults are sensible: base model embeddings win if the token exists there, single-model tokens use that model, otherwise it averages. You can override any of this per-token.

### Chat Template

```yaml
chat_template: "auto"    # picks the most common template from inputs
# or: "alpaca", "chatml", "llama3", "mistral", "exaone"
# or: a raw Jinja2 template string
```

### Examples

Check [`examples/`](examples/) for real configs.

## Merge Methods

| Method | `merge_method` | Multi-Model | Needs Base |
|---|---|---|---|
| Linear (Model Soups) | `linear` | ✅ | ❌ |
| SLERP | `slerp` | ✅* | ✅ |
| Nearswap | `nearswap` | ❌ | ✅ |
| Task Arithmetic | `task_arithmetic` | ✅ | ✅ |
| TIES | `ties` | ✅ | ✅ |
| DARE + TIES | `dare_ties` | ✅ | ✅ |
| DARE + Linear | `dare_linear` | ✅ | ✅ |
| Passthrough | `passthrough` | ❌ | ❌ |
| Model Breadcrumbs | `breadcrumbs` | ✅ | ✅ |
| Breadcrumbs + TIES | `breadcrumbs_ties` | ✅ | ✅ |
| Model Stock | `model_stock` | ✅ | ✅ |
| DELLA | `della` | ✅ | ✅ |
| DELLA + Linear | `della_linear` | ✅ | ✅ |
| SCE | `sce` | ✅ | ✅ |

\* SLERP supports two to three models.

### Linear

Weighted average. Simple, classic, effective.

- `weight` — relative weighting per tensor
- `normalize` — normalize weights across models (default: true)

### SLERP

Spherical interpolation. Supports `t` (classic SLERP, 0 = base, 1 = other) or `weight` (NuSLERP-style per-tensor weighting).

- `nuslerp_flatten` — treat tensor as flat vector vs. row/column-wise
- `nuslerp_row_wise` — SLERP row vectors instead of column vectors

### Nearswap

Interpolates between base and secondary model when similarity drops below threshold `t`.

### [Task Arithmetic](https://arxiv.org/abs/2212.04089)

Subtract base model → get "task vectors" → merge them linearly → add base back. Great for models fine-tuned from a common ancestor. Also the mental model behind most of the fancier methods.

### [TIES](https://arxiv.org/abs/2306.01708)

Task arithmetic + sparsification + sign consensus. Lets you merge more models without them stepping on each other.

- `density` — fraction of task vector weights to keep

### [DARE](https://arxiv.org/abs/2311.03099)

Random pruning with rescaling, instead of TIES's magnitude-based sparsification. Works with TIES sign consensus (`dare_ties`) or without (`dare_linear`).

### Passthrough

No-op. Passes tensors through unchanged. Useful for layer-stacking / frankenmerging where you only have one input per slice.

### [Model Breadcrumbs](https://arxiv.org/abs/2312.06795)

Drops both tiny and huge differences from base. Works with (`breadcrumbs_ties`) or without (`breadcrumbs`) TIES.

- `density` — fraction of weights to keep
- `gamma` — fraction of largest-magnitude differences to remove (paper's β)
- Defaults: `density: 0.9`, `gamma: 0.01`

### [Model Stock](https://arxiv.org/abs/2403.19522)

Geometric trick to compute good linear weights. Needs at least three models including a base.

### [DELLA](https://arxiv.org/abs/2406.11617)

Adaptive pruning based on magnitude ranking — keeps important changes, drops the rest. Like DARE but smarter about what it prunes.

- `density` — fraction of weights to keep
- `epsilon` — spread of drop probabilities (range: `density ± epsilon`)
- `lambda` — scaling factor for merged deltas

### [SCE](https://arxiv.org/abs/2408.07990)

Selects high-variance elements, computes matrix-level weights, erases minority contributions.

- `select_topk` — fraction of high-variance elements to retain

## LoRA Extraction

Extract PEFT-compatible LoRA adapters from finetuned models:

```sh
mergekitty-extract-lora finetuned_model base_model output_path --rank=32
```

## MoE Merging

Merge dense models into a Mixture of Experts with `mergekitty-moe`. See the [MoE docs](docs/moe.md).

## Development

Uses Hatch + uv:

```sh
uv tool install hatch
hatch test              # run tests
hatch run lint          # ruff linting
hatch run format        # ruff formatting
hatch run mergekitty-yaml examples/bio-merge.yml ./bio-merge --cuda
```

## Citation

If you use mergekitty in research, please cite the [original mergekit paper](https://aclanthology.org/2024.emnlp-industry.36/):

```bibtex
@inproceedings{goddard-etal-2024-arcees,
    title = "Arcee{'}s {M}erge{K}it: A Toolkit for Merging Large Language Models",
    author = "Goddard, Charles  and
      Siriwardhana, Shamane  and
      Ehghaghi, Malikeh  and
      Meyers, Luke  and
      Karpukhin, Vladimir  and
      Benedict, Brian  and
      McQuade, Mark  and
      Solawetz, Jacob",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = nov,
    year = "2024",
    pages = "477--485",
    url = "https://aclanthology.org/2024.emnlp-industry.36",
}
```
