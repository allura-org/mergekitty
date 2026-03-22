from common import make_qwen2vl_picollama, make_tokenizer, run_and_check_merge
from transformers import AutoConfig

from mergekitty.architecture import get_architecture_info
from mergekitty.config import (
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)


class TestQwen2VLArchitecture:
    def test_qwen2_vl_architecture_info_expands_vision_and_text_weights(
        self, tmp_path
    ):
        model_path = make_qwen2vl_picollama(tmp_path / "qwen2vl")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)

        cfg = AutoConfig.from_pretrained(model_path)
        arch_info = get_architecture_info(cfg)
        weights = arch_info.all_weights(cfg)
        names = {weight.name for weight in weights}

        assert arch_info.name() == "qwen2_vl"
        assert arch_info.num_layers(cfg) == 2
        assert len(weights) == 58
        assert "visual.blocks.1.attn.qkv.weight" in names
        assert "visual.merger.mlp.2.bias" in names
        assert "model.layers.1.self_attn.q_proj.weight" in names
        assert "model.norm.weight" in names
        assert "lm_head.weight" in names

    def test_qwen2_vl_passthrough_stack_updates_nested_text_layer_count(
        self, tmp_path
    ):
        model_path = make_qwen2vl_picollama(tmp_path / "qwen2vl")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)

        config = MergeConfiguration(
            merge_method="passthrough",
            slices=[
                OutputSliceDefinition(
                    sources=[
                        InputSliceDefinition(model=model_path, layer_range=[0, 2]),
                    ]
                ),
                OutputSliceDefinition(
                    sources=[
                        InputSliceDefinition(model=model_path, layer_range=[0, 2]),
                    ]
                ),
            ],
            dtype="bfloat16",
        )

        def _check_config_layers(path: str):
            out_cfg = AutoConfig.from_pretrained(path)
            assert out_cfg.text_config.num_hidden_layers == 4

        run_and_check_merge(config, validate=_check_config_layers)
