# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025 Allura-org
import os
import tempfile
from typing import Callable, Optional, List, Union
import json

from transformers import (
    AutoConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
    LlamaTokenizerFast,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
import tokenizers

from mergekitty.architecture import get_architecture_info
from mergekitty.config import MergeConfiguration
from mergekitty.io.lazy_tensor_loader import LazyTensorLoader, ShardedTensorIndex
from mergekitty.merge import MergeOptions, run_merge


def make_tokenizer(
    vocab_size: int, added_tokens: List[Union[str, tokenizers.AddedToken]]
) -> PreTrainedTokenizerBase:
    tokens = ["<unk>", "<s>", "</s>"] + [f"_tok_{idx}" for idx in range(3, vocab_size)]
    tokens = tokens[:vocab_size]
    tok_data = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": dict(zip(tokens, range(vocab_size))),
            "merges": [],
        },
        "added_tokens": [],
    }
    tok = tokenizers.Tokenizer.from_str(json.dumps(tok_data))
    with tempfile.TemporaryDirectory() as p:
        tok_path = os.path.join(p, "tokenizer.json")
        tok.save(tok_path)
        res = LlamaTokenizerFast(tokenizer_file=tok_path)

    res.add_tokens(added_tokens)
    return res


def run_and_check_merge(
    config: MergeConfiguration,
    check_nan: bool = True,
    check_tensors: bool = True,
    validate: Optional[Callable[[str], None]] = None,
    index_json_name: Optional[str] = None,
):
    if index_json_name is None:
        index_json_name = "model.safetensors.index.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        run_merge(config, out_path=tmpdir, options=MergeOptions())
        assert os.path.exists(os.path.join(tmpdir, index_json_name)), (
            "No index file for merge"
        )
        assert os.path.exists(os.path.join(tmpdir, "config.json")), (
            "No config json produced by merge"
        )

        if check_nan:
            # check for NaN in output
            loader = LazyTensorLoader.from_disk(tmpdir, lazy_unpickle=False)
            tp = loader.index.tensor_paths
            sorted_tensors = sorted(tp.keys(), key=lambda k: tp[k])
            for tensor_name in sorted_tensors:
                tensor = loader.get_tensor(tensor_name)
                has_nan = tensor.view(-1).isnan().any()
                assert not has_nan, "Output contains NaN"

        if check_tensors:
            config = AutoConfig.from_pretrained(tmpdir)
            arch_info = get_architecture_info(config)

            index = ShardedTensorIndex.from_disk(tmpdir)
            for weight_info in arch_info.all_weights(config):
                if weight_info.optional:
                    continue
                if weight_info.name not in index.tensor_paths and not any(
                    a in index.tensor_paths for a in weight_info.aliases
                ):
                    raise RuntimeError(f"Output missing tensor {weight_info.name}")

        if validate:
            validate(tmpdir)


def make_picollama(path: str, vocab_size: int = 64):
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=16,
        num_hidden_layers=2,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


def make_qwen3moe_picollama(path: str, vocab_size: int = 64):
    cfg = Qwen3MoeConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_experts=2,
        num_experts_per_tok=2,
    )
    model = Qwen3MoeForCausalLM(cfg)
    model.save_pretrained(path, safe_serialization=True)
    return str(path)
