import contextlib

import torch

from mergekitty.io import LazyTensorLoader
from mergekitty.io.loader import LazyPickleLoader


class TestLazyUnpickle:
    def test_lazy_unpickle(self, tmp_path):
        data = {
            "a": torch.tensor([1, 2, 3]),
            "b": torch.tensor([4, 5, 6]),
        }
        path = tmp_path / "pytorch_model.bin"
        torch.save(data, path)
        loader = LazyTensorLoader.from_disk(tmp_path)
        for name in data:
            assert name in loader.index.tensor_paths
            tensor = loader.get_tensor(name)
            assert torch.equal(tensor, data[name])

    def test_lazy_pickle_loader_handles_eager_tensor_entries(self):
        loader = LazyPickleLoader.__new__(LazyPickleLoader)
        loader.zip_reader = None
        loader.device = "cpu"
        loader.index = {"a": torch.tensor([1, 2, 3])}

        tensor = loader.get_tensor("a")

        assert torch.equal(tensor, torch.tensor([1, 2, 3]))

    def test_lazy_pickle_loader_forces_weights_only_false(self, monkeypatch):
        recorded = {}

        monkeypatch.setattr(
            "mergekitty.io.loader.TorchArchiveReader", lambda path: path
        )
        monkeypatch.setattr(
            "mergekitty.io.loader.torch_lazy_load",
            contextlib.nullcontext,
        )

        def fake_torch_load(path, **kwargs):
            recorded["path"] = path
            recorded["kwargs"] = kwargs
            return {}

        monkeypatch.setattr("mergekitty.io.loader.torch.load", fake_torch_load)

        loader = LazyPickleLoader("fake-model.bin")

        assert loader.index == {}
        assert recorded["path"] == "fake-model.bin"
        assert recorded["kwargs"]["weights_only"] is False
