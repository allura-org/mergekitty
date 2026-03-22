from concurrent.futures import ThreadPoolExecutor
import threading

import torch

from mergekitty.io.lazy_tensor_loader import LazyTensorLoader, ShardedTensorIndex
from mergekitty.io.loader import TensorLoader


class BlockingShardLoader:
    def __init__(
        self,
        *,
        started: set[tuple[int, str, str]],
        started_lock: threading.Lock,
        ready: threading.Event,
        release: threading.Event,
        shard_file: str,
        keys: tuple[str, ...],
    ):
        self._started = started
        self._started_lock = started_lock
        self._ready = ready
        self._release = release
        self._shard_file = shard_file
        self._keys = keys

    def keys(self):
        return self._keys

    def get_tensor(self, key: str) -> torch.Tensor:
        with self._started_lock:
            self._started.add((threading.get_ident(), self._shard_file, key))
            if len(self._started) >= 2:
                self._ready.set()

        assert self._release.wait(timeout=5), "Timed out waiting to release reads"
        return torch.tensor([ord(key[0])], dtype=torch.int64)


class CountingShardLoader:
    def __init__(self, keys: tuple[str, ...], value: int):
        self._keys = keys
        self._value = value

    def keys(self):
        return self._keys

    def get_tensor(self, key: str) -> torch.Tensor:
        return torch.tensor([self._value], dtype=torch.int64)


def test_concurrent_reads_on_same_loader_do_not_serialize(monkeypatch):
    started: set[tuple[int, str, str]] = set()
    started_lock = threading.Lock()
    ready = threading.Event()
    release = threading.Event()

    def fake_get(cls, shard_path, use_lazy_unpickle=False, device=None):
        return BlockingShardLoader(
            started=started,
            started_lock=started_lock,
            ready=ready,
            release=release,
            shard_file="model.safetensors",
            keys=("a", "b"),
        )

    monkeypatch.setattr(TensorLoader, "get", classmethod(fake_get))

    loader = LazyTensorLoader(
        ShardedTensorIndex(
            base_path="/tmp",
            is_safetensors=True,
            tensor_paths={"a": "model.safetensors", "b": "model.safetensors"},
            shards=[],
        )
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(loader.get_tensor, "a")
        future_b = pool.submit(loader.get_tensor, "b")

        assert ready.wait(timeout=5), "Concurrent reads were serialized"
        release.set()

        assert torch.equal(future_a.result(timeout=5), torch.tensor([ord("a")]))
        assert torch.equal(future_b.result(timeout=5), torch.tensor([ord("b")]))


def test_reuses_shard_loader_within_a_thread(monkeypatch):
    open_count = 0

    def fake_get(cls, shard_path, use_lazy_unpickle=False, device=None):
        nonlocal open_count
        open_count += 1
        return CountingShardLoader(keys=("a", "b"), value=open_count)

    monkeypatch.setattr(TensorLoader, "get", classmethod(fake_get))

    loader = LazyTensorLoader(
        ShardedTensorIndex(
            base_path="/tmp",
            is_safetensors=True,
            tensor_paths={"a": "model.safetensors", "b": "model.safetensors"},
            shards=[],
        )
    )

    first = loader.get_tensor("a")
    second = loader.get_tensor("b")

    assert open_count == 1
    assert torch.equal(first, torch.tensor([1]))
    assert torch.equal(second, torch.tensor([1]))


def test_flush_invalidates_thread_local_state(monkeypatch):
    open_count = 0

    def fake_get(cls, shard_path, use_lazy_unpickle=False, device=None):
        nonlocal open_count
        open_count += 1
        return CountingShardLoader(keys=("a",), value=open_count)

    monkeypatch.setattr(TensorLoader, "get", classmethod(fake_get))

    loader = LazyTensorLoader(
        ShardedTensorIndex(
            base_path="/tmp",
            is_safetensors=True,
            tensor_paths={"a": "model.safetensors"},
            shards=[],
        )
    )

    assert torch.equal(loader.get_tensor("a"), torch.tensor([1]))
    loader.flush()
    assert torch.equal(loader.get_tensor("a"), torch.tensor([2]))
    assert open_count == 2
