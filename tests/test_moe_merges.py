import pytest
from common import make_qwen3moe_picollama, make_tokenizer, run_and_check_merge
from transformers import AutoConfig

from mergekitty.config import (
    InputModelDefinition,
    MergeConfiguration,
)
from mergekitty.io import LazyTensorLoader


@pytest.fixture(scope="session")
def moe_model_a(tmp_path_factory):
    model_path = make_qwen3moe_picollama(tmp_path_factory.mktemp("moe_model_a"))
    make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)
    return model_path


@pytest.fixture(scope="session")
def moe_model_b(tmp_path_factory):
    model_path = make_qwen3moe_picollama(tmp_path_factory.mktemp("moe_model_b"))
    make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)
    return model_path


class TestMoEMerges:
    def test_linear_merge(self, moe_model_a, moe_model_b):
        config = MergeConfiguration(
            merge_method="linear",
            models=[
                InputModelDefinition(model=moe_model_a, parameters={"weight": 0.5}),
                InputModelDefinition(model=moe_model_b, parameters={"weight": 0.5}),
            ],
        )
        run_and_check_merge(config)

    def test_expert_merge_validation(self, moe_model_a, moe_model_b):
        def validate_experts(path: str):
            loader = LazyTensorLoader.from_disk(path)
            config = AutoConfig.from_pretrained(path)

            print(path)

            # Check that expert weights exist and are properly merged
            for layer_idx in range(config.num_hidden_layers):
                for expert_idx in range(config.num_experts):
                    expert_weights = [
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                    ]

                    for weight_name in expert_weights:
                        assert weight_name in loader.index.tensor_paths, (
                            f"Missing expert weight {weight_name}"
                        )
                        tensor = loader.get_tensor(weight_name)
                        assert not tensor.isnan().any(), f"NaN values in {weight_name}"
                        assert tensor.numel() > 0, f"Empty tensor for {weight_name}"

                # Check gate weights
                gate_weight = f"model.layers.{layer_idx}.mlp.gate.weight"
                assert gate_weight in loader.index.tensor_paths, (
                    f"Missing gate weight {gate_weight}"
                )

        config = MergeConfiguration(
            merge_method="linear",
            models=[
                InputModelDefinition(model=moe_model_a, parameters={"weight": 0.5}),
                InputModelDefinition(model=moe_model_b, parameters={"weight": 0.5}),
            ],
        )
        run_and_check_merge(config, validate=validate_experts)
