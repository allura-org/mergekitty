import os
import tempfile
import threading

import pytest
import torch

from mergekitty.io import TensorWriter


class TestTensorWriter:
    def test_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, safe_serialization=True)
            writer.save_tensor("steve", torch.randn(4))
            writer.finalize()

            assert os.path.exists(os.path.join(d, "model-00001-of-00001.safetensors"))
            assert os.path.exists(os.path.join(d, "model.safetensors.index.json"))

    def test_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, safe_serialization=False)
            writer.save_tensor("timothan", torch.randn(4))
            writer.finalize()

            assert os.path.exists(os.path.join(d, "pytorch_model-00001-of-00001.bin"))
            assert os.path.exists(os.path.join(d, "pytorch_model.bin.index.json"))

    def test_duplicate_tensor(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, safe_serialization=True)
            jim = torch.randn(4)
            writer.save_tensor("jim", jim)
            writer.save_tensor("jimbo", jim)
            writer.finalize()

            assert os.path.exists(os.path.join(d, "model-00001-of-00001.safetensors"))
            assert os.path.exists(os.path.join(d, "model.safetensors.index.json"))

    def test_flush_is_async(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, max_shard_size=40, safe_serialization=True)
            started = threading.Event()
            release = threading.Event()
            errors = []

            def fake_write_shard_file(shard_name, shard):
                started.set()
                assert release.wait(timeout=5)
                with open(os.path.join(d, shard_name), "wb"):
                    pass

            writer._write_shard_file = fake_write_shard_file

            tensor = torch.randn(8)
            writer.save_tensor("a", tensor)

            finished = threading.Event()

            def _save_second_tensor():
                try:
                    writer.save_tensor("b", tensor)
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)
                finally:
                    finished.set()

            thread = threading.Thread(target=_save_second_tensor, daemon=True)
            thread.start()

            assert started.wait(timeout=5), "Shard write did not start"
            assert finished.wait(timeout=1), "save_tensor blocked on shard write"
            assert not errors

            release.set()
            thread.join(timeout=5)
            assert not thread.is_alive()
            writer.finalize()

    def test_background_write_failure_surfaces(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, max_shard_size=40, safe_serialization=True)

            def fake_write_shard_file(shard_name, shard):
                raise OSError("disk full")

            writer._write_shard_file = fake_write_shard_file

            tensor = torch.randn(8)
            writer.save_tensor("a", tensor)
            writer.save_tensor("b", tensor)

            with pytest.raises(RuntimeError, match="Background shard write failed"):
                writer.finalize()
