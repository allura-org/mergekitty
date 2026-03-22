from types import SimpleNamespace

from common import (
    make_mistral3_picollama,
    make_qwen2vl_picollama,
    make_tokenizer,
    run_and_check_merge,
)
from transformers import AutoConfig, Qwen2Config, Qwen2ForCausalLM

from mergekitty.architecture import get_architecture_info
from mergekitty.config import (
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)
from mergekitty.merge import _model_out_config


class TestQwen2VLArchitecture:
    def test_qwen2_vl_architecture_info_expands_vision_and_text_weights(self, tmp_path):
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

    def test_qwen2_vl_passthrough_stack_updates_nested_text_layer_count(self, tmp_path):
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


class TestMistral3Architecture:
    def test_mistral3_architecture_info_expands_vision_and_text_weights(
        self, tmp_path
    ):
        model_path = make_mistral3_picollama(tmp_path / "mistral3")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)

        cfg = AutoConfig.from_pretrained(model_path)
        arch_info = get_architecture_info(cfg)
        weights = arch_info.all_weights(cfg)
        names = {weight.name for weight in weights}

        assert arch_info.name() == "mistral3"
        assert arch_info.definition.match_model_type == "mistral"
        assert arch_info.num_layers(cfg) == 2
        assert len(weights) == 47
        assert "vision_tower.transformer.layers.1.attention.q_proj.weight" in names
        assert "multi_modal_projector.patch_merger.merging_layer.weight" in names
        assert "language_model.model.layers.1.self_attn.q_proj.weight" in names
        assert "language_model.model.norm.weight" in names
        assert "language_model.lm_head.weight" in names

    def test_mistral3_passthrough_stack_updates_nested_text_layer_count(
        self, tmp_path
    ):
        model_path = make_mistral3_picollama(tmp_path / "mistral3")
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

    def test_mistral3_architecture_info_can_match_nested_text_model_type(self):
        cfg = SimpleNamespace(
            architectures=["Mistral3ForConditionalGeneration"],
            model_type="mistral3",
            text_config=SimpleNamespace(model_type="ministral3", num_hidden_layers=2),
            vision_config=SimpleNamespace(num_hidden_layers=2),
        )

        arch_info = get_architecture_info(cfg)
        weights = arch_info.all_weights(cfg)
        names = {weight.name for weight in weights}

        assert arch_info.name() == "mistral3"
        assert arch_info.definition.match_model_type == "ministral3"
        assert arch_info.num_layers(cfg) == 2
        assert "vision_tower.transformer.layers.1.attention.q_proj.weight" in names
        assert "language_model.model.layers.1.self_attn.q_proj.weight" in names


class TestSliceConfigMetadata:
    def test_passthrough_slice_uses_selected_layer_types_in_output_config(
        self, tmp_path
    ):
        model_path = tmp_path / "qwen2"
        cfg = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            use_sliding_window=True,
            max_window_layers=2,
        )
        model = Qwen2ForCausalLM(cfg)
        model.save_pretrained(model_path, safe_serialization=True)
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)

        merge_config = MergeConfiguration(
            merge_method="passthrough",
            slices=[
                OutputSliceDefinition(
                    sources=[
                        InputSliceDefinition(model=str(model_path), layer_range=[2, 4]),
                    ]
                )
            ],
            dtype="bfloat16",
        )

        arch_info = get_architecture_info(AutoConfig.from_pretrained(model_path))
        out_cfg = _model_out_config(merge_config, arch_info)

        assert out_cfg.num_hidden_layers == 2
        assert out_cfg.layer_types == ["sliding_attention", "sliding_attention"]
