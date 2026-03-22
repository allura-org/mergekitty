# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025-2026 Allura-org
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import json
import logging
import os
import queue
import threading
from typing import Dict, Optional, Tuple

import safetensors
import torch


class TensorWriter:
    out_path: str
    max_shard_size: int
    write_queue_depth: int
    shards_written: int
    weight_map = Dict[str, str]
    current_shard: Dict[str, torch.Tensor]
    current_shard_size: int
    total_size: int
    safe_serialization: bool
    _pending_writes: "queue.Queue[Tuple[str, Dict[str, torch.Tensor]] | object]"
    _writer_thread: threading.Thread
    _writer_error: Optional[BaseException]
    _STOP_SENTINEL = object()

    def __init__(
        self,
        out_path: str,
        max_shard_size: int = 1000 * 1000 * 1000 * 5,
        write_queue_depth: int = 1,
        safe_serialization: bool = True,
    ) -> None:
        if write_queue_depth < 1:
            raise ValueError("write_queue_depth must be at least 1")

        os.makedirs(out_path, exist_ok=True)

        self.out_path = out_path
        self.max_shard_size = max_shard_size
        self.write_queue_depth = write_queue_depth
        self.safe_serialization = safe_serialization
        self.shards_written = 0
        self.weight_map = {}
        self.current_shard = {}
        self.current_shard_size = 0
        self.total_size = 0
        self._lock = threading.RLock()
        self._pending_writes = queue.Queue(maxsize=write_queue_depth)
        self._writer_error = None
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="mergekitty-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def save_tensor(self, name: str, tensor: torch.Tensor, clone: bool = False):
        shard_to_write = None
        with self._lock:
            self._raise_writer_error_locked()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            tensor_size = tensor.numel() * tensor.element_size()
            if (
                self.current_shard
                and self.current_shard_size + tensor_size > self.max_shard_size
            ):
                shard_to_write = self._rotate_shard_locked()

            if clone:
                tensor = tensor.clone()

            self.current_shard[name] = tensor
            self.total_size += tensor_size
            self.current_shard_size += tensor_size
        if shard_to_write is not None:
            self._enqueue_write(shard_to_write)

    def flush_current_shard(self):
        shard_to_write = None
        with self._lock:
            self._raise_writer_error_locked()
            shard_to_write = self._rotate_shard_locked()
        if shard_to_write is not None:
            self._enqueue_write(shard_to_write)

    def finalize(self):
        self.flush_current_shard()
        self._enqueue_write(self._STOP_SENTINEL)
        self._writer_thread.join()

        with self._lock:
            self._raise_writer_error_locked()

            logging.info("Finalizing shard names")

            prefix, extension = self._get_name_components()

            # standardize shard names to hf format
            total_shards = self.shards_written
            name_remap = {}
            for idx in range(total_shards):
                name_remap[f"{prefix}-{idx + 1}.{extension}"] = (
                    f"{prefix}-{idx + 1:05d}-of-{total_shards:05d}.{extension}"
                )

            for old_name, new_name in name_remap.items():
                os.rename(
                    os.path.join(self.out_path, old_name),
                    os.path.join(self.out_path, new_name),
                )

            for key in self.weight_map:
                self.weight_map[key] = name_remap[self.weight_map[key]]

            with open(
                os.path.join(self.out_path, f"{prefix}.{extension}.index.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "metadata": {
                            "mergekitty_version": "0.0.7",
                            "total_size": self.total_size,
                        },
                        "weight_map": self.weight_map,
                    },
                    file,
                )

    def _get_name_components(self):
        if self.safe_serialization:
            return "model", "safetensors"
        return "pytorch_model", "bin"

    def _writer_loop(self):
        while True:
            item = self._pending_writes.get()
            try:
                if item is self._STOP_SENTINEL:
                    return

                shard_name, shard = item
                logging.info("Writing shard %s to disk", shard_name)
                self._write_shard_file(shard_name, shard)
            except BaseException as exc:  # pragma: no cover - surfaced via public API
                with self._lock:
                    if self._writer_error is None:
                        self._writer_error = exc
                return
            finally:
                self._pending_writes.task_done()

    def _enqueue_write(self, item: Tuple[str, Dict[str, torch.Tensor]] | object):
        while True:
            with self._lock:
                self._raise_writer_error_locked()
            try:
                self._pending_writes.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _raise_writer_error_locked(self):
        if self._writer_error is not None:
            raise RuntimeError("Background shard write failed") from self._writer_error

    def _rotate_shard_locked(self) -> Optional[Tuple[str, Dict[str, torch.Tensor]]]:
        if not self.current_shard:
            return None

        prefix, extension = self._get_name_components()
        shard_name = f"{prefix}-{self.shards_written + 1}.{extension}"
        shard = self.current_shard

        for key in shard:
            self.weight_map[key] = shard_name

        self.current_shard = {}
        self.current_shard_size = 0
        self.shards_written = self.shards_written + 1
        return shard_name, shard

    def _write_shard_file(self, shard_name: str, shard: Dict[str, torch.Tensor]):
        shard_path = os.path.join(self.out_path, shard_name)
        if self.safe_serialization:
            self._save_st(shard_path, shard)
        else:
            torch.save(shard, shard_path)

    def _save_st(self, shard_path: str, shard: Dict[str, torch.Tensor]):
        def _do_save():
            safetensors.torch.save_file(
                shard,
                shard_path,
                metadata={"format": "pt"},
            )

        try:
            _do_save()
        except RuntimeError as e:
            if (
                len(e.args) > 0
                and isinstance(e.args[0], str)
                and "share memory" in e.args[0]
            ):
                logging.warning(
                    "Your model has duplicated tensors but the --clone-tensors "
                    "flag is not set."
                )
                shard = {key: shard[key].clone() for key in shard}
                _do_save()
            else:
                raise
