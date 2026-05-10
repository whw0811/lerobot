#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read-side caches for LeRobotDataset."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
from PIL import Image as PILImage

from lerobot.utils.io_utils import write_json

CACHE_VERSION = 1
IMAGE_CACHE_DIR = "image_cache"


def safe_cache_key(key: str) -> str:
    """Return a filesystem-safe cache filename stem for a feature key."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return safe.replace("/", "_")


def compute_index_hash(hf_dataset: datasets.Dataset) -> str:
    """Hash the ordered dataset index column for cache validation."""
    try:
        indices = hf_dataset.data.column("index").to_numpy()
    except Exception:
        values = hf_dataset["index"]
        indices = np.asarray(
            [int(value.item() if isinstance(value, torch.Tensor) else value) for value in values],
            dtype=np.int64,
        )

    indices = np.asarray(indices, dtype=np.int64)
    hasher = hashlib.sha256()
    hasher.update(np.asarray([len(indices)], dtype=np.int64).tobytes())
    hasher.update(indices.tobytes())
    return hasher.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def image_to_uint8_chw(value: Any, *, key: str, row_idx: int) -> np.ndarray:
    """Convert an HF image value to uint8 CHW for memmap storage."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(
                f"Image cache expects 3D image for key {key!r} at row {row_idx}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] in (1, 3):
            chw = tensor
        elif tensor.shape[-1] in (1, 3):
            chw = tensor.permute(2, 0, 1)
        else:
            raise ValueError(
                f"Image cache cannot infer channel axis for key {key!r} at row {row_idx}, "
                f"got {tuple(tensor.shape)}"
            )
        if chw.dtype == torch.uint8:
            return chw.numpy()
        return (chw.float().clamp(0, 1) * 255).round().to(torch.uint8).numpy()

    if isinstance(value, PILImage.Image):
        array = np.asarray(value.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(value)

    if array.ndim != 3:
        raise ValueError(f"Image cache expects 3D image for key {key!r} at row {row_idx}, got {array.shape}")
    if array.shape[0] in (1, 3):
        chw = array
    elif array.shape[-1] in (1, 3):
        chw = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(
            f"Image cache cannot infer channel axis for key {key!r} at row {row_idx}, got {array.shape}"
        )
    if np.issubdtype(chw.dtype, np.floating):
        chw = np.clip(chw, 0, 1) * 255
    return np.rint(chw).astype(np.uint8, copy=False)


class ImageMemmapCache:
    """Memmap-backed cache for image-backed visual columns."""

    def __init__(
        self,
        root: Path,
        hf_dataset: datasets.Dataset,
        image_keys: list[str],
        features: dict[str, dict],
        *,
        return_uint8: bool,
    ) -> None:
        self.root = root
        self.hf_dataset = hf_dataset
        self.image_keys = image_keys
        self.features = features
        self.return_uint8 = return_uint8
        self.cache_dir = root / IMAGE_CACHE_DIR
        self.index_hash = compute_index_hash(hf_dataset)

    def get(self, key: str, relative_idx: int) -> torch.Tensor:
        mmap = self._ensure_memmap(key)
        array = np.asarray(mmap[relative_idx])
        tensor = torch.from_numpy(array.copy())
        if self.return_uint8:
            return tensor
        return tensor.float().div(255.0)

    def _paths(self, key: str) -> tuple[Path, Path]:
        safe = safe_cache_key(key)
        return self.cache_dir / f"{safe}.uint8_chw.memmap", self.cache_dir / f"{safe}.manifest.json"

    def _expected_shape(self, key: str) -> tuple[int, int, int, int]:
        if len(self.hf_dataset) == 0:
            feature_shape = tuple(self.features[key]["shape"])
            if len(feature_shape) != 3:
                raise ValueError(f"Image cache expects 3D feature shape for key {key!r}, got {feature_shape}")
            if feature_shape[0] in (1, 3):
                c, h, w = feature_shape
            elif feature_shape[-1] in (1, 3):
                h, w, c = feature_shape
            else:
                raise ValueError(f"Image cache cannot infer channel axis for key {key!r}, got {feature_shape}")
            return (0, int(c), int(h), int(w))

        first_image = image_to_uint8_chw(self.hf_dataset[key][0], key=key, row_idx=0)
        c, h, w = first_image.shape
        return (len(self.hf_dataset), int(c), int(h), int(w))

    def _manifest_is_valid(
        self, key: str, memmap_path: Path, manifest_path: Path, shape: tuple[int, ...]
    ) -> bool:
        manifest = _read_json(manifest_path)
        if manifest is None or not memmap_path.exists():
            return False
        expected_size = int(np.prod(shape)) * np.dtype(np.uint8).itemsize
        return (
            manifest.get("version") == CACHE_VERSION
            and manifest.get("key") == key
            and manifest.get("filename") == memmap_path.name
            and manifest.get("dtype") == "uint8"
            and manifest.get("layout") == "CHW"
            and tuple(manifest.get("shape", ())) == shape
            and manifest.get("num_frames") == len(self.hf_dataset)
            and manifest.get("index_hash") == self.index_hash
            and memmap_path.stat().st_size == expected_size
        )

    def _ensure_memmap(self, key: str) -> np.memmap:
        memmap_path, manifest_path = self._paths(key)
        shape = self._expected_shape(key)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self._manifest_is_valid(key, memmap_path, manifest_path, shape):
            self._build(key, memmap_path, manifest_path, shape)
        return np.memmap(memmap_path, mode="r", dtype=np.uint8, shape=shape)

    def _build(self, key: str, memmap_path: Path, manifest_path: Path, shape: tuple[int, ...]) -> None:
        if manifest_path.exists():
            manifest_path.unlink()

        mmap = np.memmap(memmap_path, mode="w+", dtype=np.uint8, shape=shape)
        try:
            column = self.hf_dataset[key]
            for row_idx in range(len(self.hf_dataset)):
                mmap[row_idx] = image_to_uint8_chw(column[row_idx], key=key, row_idx=row_idx)
            mmap.flush()
        finally:
            del mmap

        write_json(
            {
                "version": CACHE_VERSION,
                "key": key,
                "filename": memmap_path.name,
                "dtype": "uint8",
                "layout": "CHW",
                "shape": list(shape),
                "feature_shape": list(self.features[key]["shape"]),
                "num_frames": len(self.hf_dataset),
                "index_hash": self.index_hash,
            },
            manifest_path,
        )


class DeltaColumnCache:
    """In-memory cache for numeric columns used by delta timestamp windows."""

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        delta_indices: dict[str, list[int]] | None,
        features: dict[str, dict],
        camera_keys: list[str],
    ) -> None:
        self._columns: dict[str, torch.Tensor] = {}
        if delta_indices is None:
            return

        for key in delta_indices:
            if key in camera_keys or key not in features or key not in hf_dataset.features:
                continue
            dtype = features[key].get("dtype")
            if dtype in {"image", "video", "string"}:
                continue
            try:
                values = hf_dataset[key]
                tensors = [value if isinstance(value, torch.Tensor) else torch.tensor(value) for value in values]
                self._columns[key] = torch.stack(tensors)
            except (TypeError, ValueError, RuntimeError, KeyError):
                continue

    def get(self, key: str, relative_indices: list[int]) -> torch.Tensor | None:
        tensor = self._columns.get(key)
        if tensor is None:
            return None
        return tensor[torch.as_tensor(relative_indices, dtype=torch.long)]
