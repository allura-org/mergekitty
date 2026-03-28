from types import SimpleNamespace

from common import (
    make_gemma3_picollama,
    make_mistral3_picollama,
    make_qwen2vl_picollama,
    make_qwen3_5_picollama,
    make_tokenizer,
    run_and_check_merge,
)
from transformers import AutoConfig, Qwen2Config, Qwen2ForCausalLM

from mergekitty.architecture import get_architecture_info
from mergekitty.config import (
    InputModelDefinition,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)
from mergekitty.merge import _model_out_config
from mergekitty.options import MergeOptions
from mergekitty.plan import MergePlanner


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
    def test_mistral3_architecture_info_expands_vision_and_text_weights(self, tmp_path):
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

    def test_mistral3_passthrough_stack_updates_nested_text_layer_count(self, tmp_path):
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


class TestQwen3_5Architecture:
    def test_qwen3_5_architecture_info_expands_vision_and_text_weights(self, tmp_path):
        model_path = make_qwen3_5_picollama(tmp_path / "qwen3_5")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)

        cfg = AutoConfig.from_pretrained(model_path)
        arch_info = get_architecture_info(cfg)
        weights = arch_info.all_weights(cfg)
        names = {weight.name for weight in weights}

        assert arch_info.name() == "qwen3_5"
        assert arch_info.num_layers(cfg) == 4
        assert len(weights) == 89
        assert "model.visual.blocks.1.attn.qkv.weight" in names
        assert "model.visual.merger.linear_fc2.bias" in names
        assert "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" in names
        assert "model.language_model.layers.3.self_attn.q_proj.weight" in names
        assert "model.language_model.layers.0.self_attn.q_proj.weight" not in names
        assert (
            "model.language_model.layers.3.linear_attn.in_proj_qkv.weight" not in names
        )
        assert "model.language_model.norm.weight" in names
        assert "lm_head.weight" in names

    def test_qwen3_5_models_input_normalizes_into_full_layer_slice(self, tmp_path):
        model_path = make_qwen3_5_picollama(tmp_path / "qwen3_5")
        cfg = AutoConfig.from_pretrained(model_path)
        arch_info = get_architecture_info(cfg)
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[InputModelDefinition(model=model_path)],
            dtype="bfloat16",
        )
        planner = MergePlanner(
            config=config,
            arch_info=arch_info,
            options=MergeOptions(),
            out_model_config=_model_out_config(config, arch_info),
        )

        planner.normalize_config()

        assert config.models is None
        assert config.slices is not None
        assert len(config.slices) == 1
        assert len(config.slices[0].sources) == 1
        assert str(config.slices[0].sources[0].model) == model_path
        assert config.slices[0].sources[0].layer_range == (0, 4)

    def test_qwen3_5_passthrough_updates_nested_text_layer_types(self, tmp_path):
        model_path = make_qwen3_5_picollama(tmp_path / "qwen3_5")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)

        config = MergeConfiguration(
            merge_method="passthrough",
            slices=[
                OutputSliceDefinition(
                    sources=[
                        InputSliceDefinition(model=model_path, layer_range=[2, 4]),
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
            assert out_cfg.text_config.layer_types == [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
            ]

        run_and_check_merge(config, validate=_check_config_layers)


class TestGemma3Architecture:
    def test_gemma3_architecture_info_expands_vision_and_text_weights(self, tmp_path):
        model_path = make_gemma3_picollama(tmp_path / "gemma3")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)

        cfg = AutoConfig.from_pretrained(model_path)
        arch_info = get_architecture_info(cfg)
        weights = arch_info.all_weights(cfg)
        names = {weight.name for weight in weights}

        assert arch_info.name() == "gemma3"
        assert arch_info.num_layers(cfg) == 2
        assert len(weights) == 79
        assert (
            "vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.weight"
            in names
        )
        assert "vision_tower.vision_model.head.probe" in names
        assert "multi_modal_projector.mm_input_projection_weight" in names
        assert "language_model.model.layers.1.self_attn.q_proj.weight" in names
        assert "language_model.model.layers.1.self_attn.q_norm.weight" in names
        assert "language_model.model.norm.weight" in names
        assert "language_model.lm_head.weight" in names

    def test_gemma3_passthrough_stack_updates_nested_text_layer_count(self, tmp_path):
        model_path = make_gemma3_picollama(tmp_path / "gemma3")
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
            assert out_cfg.text_config.layer_types == [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
            ]

        run_and_check_merge(config, validate=_check_config_layers)


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


class TestApertusArchitecture:
    def test_apertus_architecture_info_expands_layer_weights(self):
        cfg = SimpleNamespace(
            architectures=["ApertusForCausalLM"],
            model_type="apertus",
            num_hidden_layers=2,
        )

        arch_info = get_architecture_info(cfg)
        weights = arch_info.all_weights(cfg)
        names = {weight.name for weight in weights}

        assert arch_info.name() == "apertus"
        assert arch_info.num_layers(cfg) == 2
        assert len(weights) == 31
        assert "model.layers.1.attention_layernorm.weight" in names
        assert "model.layers.1.feedforward_layernorm.weight" in names
        assert "model.layers.1.mlp.act_fn.alpha_n" in names
        assert "model.layers.1.mlp.act_fn.alpha_p" in names
        assert "model.layers.1.mlp.act_fn.beta" in names
        assert "model.layers.1.mlp.act_fn.eps" in names
        assert "model.layers.1.mlp.down_proj.weight" in names
        assert "model.layers.1.mlp.up_proj.weight" in names
        assert "model.layers.1.self_attn.q_norm.weight" in names
        assert "model.layers.1.self_attn.k_norm.weight" in names
        assert "model.layers.1.self_attn.q_proj.weight" in names
        assert "model.layers.1.self_attn.k_proj.weight" in names
        assert "model.layers.1.self_attn.v_proj.weight" in names
        assert "model.layers.1.self_attn.o_proj.weight" in names
        assert "model.norm.weight" in names
        assert "lm_head.weight" in names
