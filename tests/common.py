# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025-2026 Allura-org
import os
import tempfile
from typing import Callable, Optional, List, Union
import json

import pytest
from transformers import (
    AutoConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
    LlamaTokenizerFast,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
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


def make_qwen2vl_picollama(path: str, vocab_size: int = 64):
    cfg = Qwen2VLConfig(
        text_config={
            "vocab_size": vocab_size,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 2,
            "embed_dim": 32,
            "hidden_size": 32,
            "hidden_act": "gelu",
            "mlp_ratio": 2,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    model = Qwen2VLForConditionalGeneration(cfg)
    model.config.architectures = ["Qwen2VLForConditionalGeneration"]
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


def make_qwen3_5_picollama(path: str, vocab_size: int = 64):
    try:
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    except ImportError:
        pytest.skip("transformers build does not include Qwen3.5")

    cfg = Qwen3_5Config(
        text_config={
            "vocab_size": vocab_size,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "max_position_embeddings": 128,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 32,
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    model = Qwen3_5ForConditionalGeneration(cfg)
    model.config.architectures = ["Qwen3_5ForConditionalGeneration"]
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


def make_mistral3_picollama(path: str, vocab_size: int = 64):
    try:
        from transformers import Mistral3Config, Mistral3ForConditionalGeneration
    except ImportError:
        pytest.skip("transformers build does not include Mistral3")

    cfg = Mistral3Config(
        text_config={
            "model_type": "mistral",
            "vocab_size": vocab_size,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_channels": 3,
            "patch_size": 4,
            "image_size": 16,
            "hidden_act": "gelu",
        },
        image_token_index=3,
        spatial_merge_size=2,
        tie_word_embeddings=False,
    )
    model = Mistral3ForConditionalGeneration(cfg)
    model.config.architectures = ["Mistral3ForConditionalGeneration"]
    model.save_pretrained(path, safe_serialization=True)
    return str(path)
