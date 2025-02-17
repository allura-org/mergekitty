from typing import Dict, Optional

import pytest
from common import make_picollama, make_tokenizer, run_and_check_merge
from transformers import AutoConfig

from mergekitty.config import (
    InputModelDefinition,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
    ParameterSetting,
)
from mergekitty.io import LazyTensorLoader


@pytest.fixture(scope="session")
def model_a(tmp_path_factory):
    model_path = make_picollama(tmp_path_factory.mktemp("model_a"))
    make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)
    return model_path


@pytest.fixture(scope="session")
def model_b(tmp_path_factory):
    model_path = make_picollama(tmp_path_factory.mktemp("model_b"))
    make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)
    return model_path


@pytest.fixture(scope="session")
def model_c(tmp_path_factory):
    model_path = make_picollama(tmp_path_factory.mktemp("model_c"))
    make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)
    return model_path


class TestBasicMerges:
    def test_gpt2_copy(self):
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[InputModelDefinition(model="gpt2")],
            dtype="bfloat16",
        )
        run_and_check_merge(config)

    def test_gpt2_stack(self):
        config = MergeConfiguration(
            merge_method="passthrough",
            slices=[
                OutputSliceDefinition(
                    sources=[InputSliceDefinition(model="gpt2", layer_range=[0, 12])]
                )
            ]
            * 2,
            dtype="bfloat16",
        )

        def _check_config_layers(p: str):
            config = AutoConfig.from_pretrained(p)
            assert config.n_layer == 24

        run_and_check_merge(config, validate=_check_config_layers)

    def test_passthrough_scale(self, model_a):
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[
                InputModelDefinition(
                    model=model_a,
                    parameters={
                        "scale": [
                            {"filter": "o_proj", "value": 0},
                            {"value": 1},
                        ]
                    },
                )
            ],
        )

        def _check_o_proj(p: str):
            loader = LazyTensorLoader.from_disk(p)
            saw_any = False
            for name in loader.index.tensor_paths:
                if "o_proj" in name:
                    param = loader.get_tensor(name)
                    assert (param == 0).all()
                    saw_any = True
                elif "lm_head" in name:
                    param = loader.get_tensor(name)
                    assert param.count_nonzero() > 0

            assert saw_any, "No o_proj parameters found"

        run_and_check_merge(config, validate=_check_o_proj)

    def test_linear_merge(self, model_a, model_b):
        config = self.two_model_config(model_a, model_b, merge_method="linear")
        run_and_check_merge(config)

    def test_slerp_merge(self, model_a, model_b):
        config = self.two_model_config(
            model_a, model_b, merge_method="slerp", base_model=model_a
        )
        config.parameters = {"t": 0.35}
        run_and_check_merge(config)

    def test_nearswap_merge(self, model_a, model_b):
        config = self.two_model_config(
            model_a, model_b, merge_method="nearswap", base_model=model_a
        )
        config.parameters = {"t": 0.0001}
        run_and_check_merge(config)

    def test_nuslerp_merges(self, model_a, model_b, model_c):
        for base_model in [None, model_c]:
            for row_wise in [False, True]:
                for flatten in [False, True]:
                    print(
                        f"Testing nuslerp with row_wise={row_wise}, flatten={flatten}, base_model={base_model}"
                    )
                    run_and_check_merge(
                        self.two_model_config(
                            model_a,
                            model_b,
                            merge_method="slerp",
                            base_model=base_model,
                            params={
                                "nuslerp_row_wise": row_wise,
                                "nuslerp_flatten": flatten,
                            },
                        )
                    )

        # test weights that sum to zero
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="slerp",
            base_model=model_c,
            params={"nuslerp_row_wise": False, "nuslerp_flatten": False},
        )
        config.models[0].parameters["weight"] = -0.5
        config.models[1].parameters["weight"] = 0.5
        run_and_check_merge(config)

    def test_task_arithmetic_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a, model_b, merge_method="task_arithmetic", base_model=model_c
        )
        run_and_check_merge(config)

    def test_breadcrumbs_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a, model_b, merge_method="breadcrumbs", base_model=model_c
        )
        run_and_check_merge(config)

    def test_ties_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="ties",
            base_model=model_c,
            params={"density": 0.3},
        )
        run_and_check_merge(config)

    def test_multislerp_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="multislerp",
            base_model=model_c,
        )
        run_and_check_merge(config)

    def test_dare_ties_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_a,
            model_b,
            merge_method="dare_ties",
            base_model=model_c,
            params={"density": 0.66},
        )
        run_and_check_merge(config)

    def test_model_stock_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_b, model_c, merge_method="model_stock", base_model=model_a
        )
        run_and_check_merge(config)

    def test_model_stock_filterwise_merge(self, model_a, model_b, model_c):
        config = self.two_model_config(
            model_b,
            model_c,
            merge_method="model_stock",
            base_model=model_a,
            params={"filter_wise": True},
        )
        run_and_check_merge(config)

    def two_model_config(
        self,
        model_a,
        model_b,
        merge_method: str,
        base_model: Optional[str] = None,
        params: Optional[Dict[str, ParameterSetting]] = None,
    ):
        config = MergeConfiguration(
            merge_method=merge_method,
            base_model=base_model,
            models=[
                InputModelDefinition(
                    model=model_a,
                    parameters={"weight": 0.6},
                ),
                InputModelDefinition(
                    model=model_b,
                    parameters={"weight": 0.4},
                ),
            ],
            dtype="bfloat16",
            parameters=params,
        )

        return config
