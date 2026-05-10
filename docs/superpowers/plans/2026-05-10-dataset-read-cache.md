# Dataset Read Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic image memmap caching and delta numeric column caching to `LeRobotDataset` reads without changing public dataset APIs.

**Architecture:** Put cache mechanics in a new `src/lerobot/datasets/cache_utils.py` module. `DatasetReader` initializes caches after loading the HF Dataset, replaces image-backed current-frame values from memmap, and serves delta windows from tensor cache before falling back to HF Dataset queries.

**Tech Stack:** Python 3.12, PyTorch, NumPy memmap, Hugging Face `datasets`, pytest, uv.

---

## File Structure

- Create `src/lerobot/datasets/cache_utils.py`
  - Owns cache path sanitization, stable `index_hash`, image conversion to/from uint8 CHW memmaps, manifest validation, and in-memory delta tensor lookup.
- Modify `src/lerobot/datasets/dataset_reader.py`
  - Imports the new cache classes.
  - Adds `_init_read_caches()`.
  - Calls cache initialization after `_build_index_mapping()`.
  - Replaces image-backed row values from cache inside `get_item()`.
  - Uses `DeltaColumnCache` in `_query_hf_dataset()` before existing fallback logic.
- Create `tests/datasets/test_dataset_read_cache.py`
  - Focused tests for image cache files, reuse, invalidation, dtype behavior, delta cache behavior, episode filters, and video exclusion.

## Task 1: Image Cache Tests

**Files:**
- Create: `tests/datasets/test_dataset_read_cache.py`
- Modify: none
- Test: `tests/datasets/test_dataset_read_cache.py`

- [ ] **Step 1: Write failing tests for image cache creation and dtype behavior**

Add the initial test module:

```python
from pathlib import Path

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _manifest_files(root: Path) -> list[Path]:
    return sorted((root / "image_cache").glob("*.manifest.json"))


def _memmap_files(root: Path) -> list[Path]:
    return sorted((root / "image_cache").glob("*.memmap"))


def test_image_cache_is_created_on_first_image_access(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=6,
        use_videos=False,
    )
    image_key = dataset.meta.image_keys[0]

    item = dataset[0]

    assert image_key in item
    assert (dataset.root / "image_cache").is_dir()
    assert len(_manifest_files(dataset.root)) == len(dataset.meta.image_keys)
    assert len(_memmap_files(dataset.root)) == len(dataset.meta.image_keys)
    assert item[image_key].dtype == torch.float32
    assert item[image_key].shape[0] == 3
    assert item[image_key].min() >= 0
    assert item[image_key].max() <= 1


def test_image_cache_respects_return_uint8(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=6,
        use_videos=False,
        return_uint8=True,
    )
    image_key = dataset.meta.image_keys[0]

    item = dataset[0]

    assert item[image_key].dtype == torch.uint8
    assert item[image_key].shape[0] == 3
```

- [ ] **Step 2: Run image cache tests and verify RED**

Run:

```bash
uv run pytest tests/datasets/test_dataset_read_cache.py::test_image_cache_is_created_on_first_image_access tests/datasets/test_dataset_read_cache.py::test_image_cache_respects_return_uint8 -q
```

Expected: FAIL because `root/image_cache/` is not created and current image reads do not use the new cache implementation.

## Task 2: Image Cache Implementation

**Files:**
- Create: `src/lerobot/datasets/cache_utils.py`
- Modify: `src/lerobot/datasets/dataset_reader.py`
- Test: `tests/datasets/test_dataset_read_cache.py`

- [ ] **Step 1: Implement cache utilities**

Create `src/lerobot/datasets/cache_utils.py` with:

```python
#!/usr/bin/env python

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
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return safe.replace("/", "_")


def compute_index_hash(hf_dataset: datasets.Dataset) -> str:
    try:
        indices = hf_dataset.data.column("index").to_numpy()
    except Exception:
        values = hf_dataset["index"]
        indices = np.asarray([int(v.item() if isinstance(v, torch.Tensor) else v) for v in values], dtype=np.int64)
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
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"Image cache expects 3D image for key {key!r} at row {row_idx}, got {tuple(tensor.shape)}")
        if tensor.shape[0] in (1, 3):
            chw = tensor
        elif tensor.shape[-1] in (1, 3):
            chw = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Image cache cannot infer channel axis for key {key!r} at row {row_idx}, got {tuple(tensor.shape)}")
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
        raise ValueError(f"Image cache cannot infer channel axis for key {key!r} at row {row_idx}, got {array.shape}")
    if np.issubdtype(chw.dtype, np.floating):
        chw = np.clip(chw, 0, 1) * 255
    return np.rint(chw).astype(np.uint8, copy=False)


class ImageMemmapCache:
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

    def contains(self, key: str) -> bool:
        return key in self.image_keys

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
        feature_shape = tuple(self.features[key]["shape"])
        if len(feature_shape) != 3:
            raise ValueError(f"Image cache expects 3D feature shape for key {key!r}, got {feature_shape}")
        if feature_shape[0] in (1, 3):
            c, h, w = feature_shape
        elif feature_shape[-1] in (1, 3):
            h, w, c = feature_shape
        else:
            raise ValueError(f"Image cache cannot infer channel axis for key {key!r}, got {feature_shape}")
        return (len(self.hf_dataset), int(c), int(h), int(w))

    def _manifest_is_valid(self, key: str, memmap_path: Path, manifest_path: Path, shape: tuple[int, ...]) -> bool:
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
        manifest = {
            "version": CACHE_VERSION,
            "key": key,
            "filename": memmap_path.name,
            "dtype": "uint8",
            "layout": "CHW",
            "shape": list(shape),
            "feature_shape": list(self.features[key]["shape"]),
            "num_frames": len(self.hf_dataset),
            "index_hash": self.index_hash,
        }
        write_json(manifest, manifest_path)


class DeltaColumnCache:
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
                tensors = [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values]
                self._columns[key] = torch.stack(tensors)
            except (TypeError, ValueError, RuntimeError, KeyError):
                continue

    def get(self, key: str, relative_indices: list[int]) -> torch.Tensor | None:
        tensor = self._columns.get(key)
        if tensor is None:
            return None
        return tensor[torch.as_tensor(relative_indices, dtype=torch.long)]
```

- [ ] **Step 2: Wire caches into `DatasetReader`**

Modify `src/lerobot/datasets/dataset_reader.py`:

```python
from .cache_utils import DeltaColumnCache, ImageMemmapCache
```

Add attributes in `__init__`:

```python
self._image_cache: ImageMemmapCache | None = None
self._delta_column_cache: DeltaColumnCache | None = None
```

After each `_build_index_mapping()` call in `try_load()` and `load_and_activate()`, call:

```python
self._init_read_caches()
```

Add the helper:

```python
def _init_read_caches(self) -> None:
    if self.hf_dataset is None:
        self._image_cache = None
        self._delta_column_cache = None
        return
    self._delta_column_cache = DeltaColumnCache(
        self.hf_dataset,
        self.delta_indices,
        self._meta.features,
        self._meta.camera_keys,
    )
    self._image_cache = ImageMemmapCache(
        self.root,
        self.hf_dataset,
        self._meta.image_keys,
        self._meta.features,
        return_uint8=self._return_uint8,
    )
```

In `get_item()`, after `item = self.hf_dataset[idx]`, replace image values:

```python
if self._image_cache is not None:
    for image_key in self._meta.image_keys:
        if image_key in item:
            item[image_key] = self._image_cache.get(image_key, idx)
```

- [ ] **Step 3: Run image cache tests and verify GREEN**

Run:

```bash
uv run pytest tests/datasets/test_dataset_read_cache.py::test_image_cache_is_created_on_first_image_access tests/datasets/test_dataset_read_cache.py::test_image_cache_respects_return_uint8 -q
```

Expected: PASS.

- [ ] **Step 4: Commit image cache baseline**

Run:

```bash
git add src/lerobot/datasets/cache_utils.py src/lerobot/datasets/dataset_reader.py tests/datasets/test_dataset_read_cache.py
git commit -m "feat: add dataset image memmap cache"
```

## Task 3: Image Cache Reuse, Invalidation, and Video Exclusion

**Files:**
- Modify: `tests/datasets/test_dataset_read_cache.py`
- Modify: `src/lerobot/datasets/cache_utils.py`
- Test: `tests/datasets/test_dataset_read_cache.py`

- [ ] **Step 1: Write failing tests for reuse, invalidation, and video exclusion**

Append:

```python
import json


def test_image_cache_reuses_existing_valid_memmap(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=6,
        use_videos=False,
    )
    _ = dataset[0]
    memmap_path = _memmap_files(dataset.root)[0]
    first_mtime = memmap_path.stat().st_mtime_ns

    reloaded = LeRobotDataset(dataset.repo_id, root=dataset.root, download_videos=False)
    _ = reloaded[0]

    assert memmap_path.stat().st_mtime_ns == first_mtime


def test_image_cache_rebuilds_when_index_hash_mismatches(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=6,
        use_videos=False,
    )
    _ = dataset[0]
    manifest_path = _manifest_files(dataset.root)[0]
    memmap_path = _memmap_files(dataset.root)[0]
    first_mtime = memmap_path.stat().st_mtime_ns

    manifest = json.loads(manifest_path.read_text())
    manifest["index_hash"] = "stale"
    manifest_path.write_text(json.dumps(manifest))

    reloaded = LeRobotDataset(dataset.repo_id, root=dataset.root, download_videos=False)
    _ = reloaded[0]

    assert memmap_path.stat().st_mtime_ns > first_mtime
    assert json.loads(manifest_path.read_text())["index_hash"] != "stale"


def test_video_backed_dataset_does_not_create_image_memmap(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=6,
        use_videos=True,
    )

    if len(dataset.meta.image_keys) == 0:
        assert not (dataset.root / "image_cache").exists()
```

- [ ] **Step 2: Run new tests and verify RED if implementation is incomplete**

Run:

```bash
uv run pytest tests/datasets/test_dataset_read_cache.py::test_image_cache_reuses_existing_valid_memmap tests/datasets/test_dataset_read_cache.py::test_image_cache_rebuilds_when_index_hash_mismatches tests/datasets/test_dataset_read_cache.py::test_video_backed_dataset_does_not_create_image_memmap -q
```

Expected: PASS if Task 2 already implements manifest validation correctly; otherwise FAIL on the missing reuse or invalidation behavior.

- [ ] **Step 3: Tighten implementation if needed**

If invalidation does not rebuild on Windows due timestamp granularity, change the test to compare manifest hash instead of mtime and ensure `_manifest_is_valid()` rejects mismatched `index_hash` exactly as shown in Task 2.

- [ ] **Step 4: Run complete image cache test file**

Run:

```bash
uv run pytest tests/datasets/test_dataset_read_cache.py -q
```

Expected: PASS for all image-cache tests present so far.

- [ ] **Step 5: Commit reuse and invalidation tests**

Run:

```bash
git add src/lerobot/datasets/cache_utils.py tests/datasets/test_dataset_read_cache.py
git commit -m "test: cover dataset image cache reuse"
```

## Task 4: Delta Column Cache Tests and Implementation

**Files:**
- Modify: `tests/datasets/test_dataset_read_cache.py`
- Modify: `src/lerobot/datasets/cache_utils.py`
- Modify: `src/lerobot/datasets/dataset_reader.py`
- Test: `tests/datasets/test_dataset_read_cache.py`

- [ ] **Step 1: Write failing delta cache tests**

Append:

```python
def test_delta_column_cache_serves_numeric_delta_queries(tmp_path, empty_lerobot_dataset_factory, monkeypatch):
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["x"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["x"]},
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False, fps=10)
    for frame_idx in range(5):
        dataset.add_frame(
            {
                "observation.state": torch.tensor([frame_idx], dtype=torch.float32),
                "action": torch.tensor([frame_idx + 100], dtype=torch.float32),
                "task": "task",
            }
        )
    dataset.save_episode()
    dataset.finalize()

    loaded = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        delta_timestamps={"observation.state": [-0.1, 0.0], "action": [0.0, 0.1]},
        tolerance_s=0.04,
    )
    calls = {"count": 0}
    original_get = loaded.reader._delta_column_cache.get

    def counted_get(key, relative_indices):
        calls["count"] += 1
        return original_get(key, relative_indices)

    monkeypatch.setattr(loaded.reader._delta_column_cache, "get", counted_get)

    item = loaded[2]

    assert calls["count"] == 2
    assert item["observation.state"].tolist() == [[1.0], [2.0]]
    assert item["action"].tolist() == [[102.0], [103.0]]


def test_delta_column_cache_preserves_episode_filter_indices(tmp_path, empty_lerobot_dataset_factory):
    features = {"observation.state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False, fps=10)
    for ep_idx in range(3):
        for frame_idx in range(5):
            dataset.add_frame(
                {
                    "observation.state": torch.tensor([ep_idx * 10 + frame_idx], dtype=torch.float32),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    loaded = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        episodes=[1],
        delta_timestamps={"observation.state": [-0.1, 0.0]},
        tolerance_s=0.04,
    )

    item = loaded[2]

    assert item["observation.state"].tolist() == [[11.0], [12.0]]
    assert item["observation.state_is_pad"].tolist() == [False, False]
```

- [ ] **Step 2: Run delta tests and verify RED**

Run:

```bash
uv run pytest tests/datasets/test_dataset_read_cache.py::test_delta_column_cache_serves_numeric_delta_queries tests/datasets/test_dataset_read_cache.py::test_delta_column_cache_preserves_episode_filter_indices -q
```

Expected: FAIL until `_query_hf_dataset()` asks `DeltaColumnCache` before the HF Dataset fallback.

- [ ] **Step 3: Implement delta cache lookup in `_query_hf_dataset()`**

Modify `_query_hf_dataset()`:

```python
def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
    """Query dataset for indices across keys, skipping video keys."""
    result: dict = {}
    for key, q_idx in query_indices.items():
        if key in self._meta.video_keys:
            continue
        relative_indices = (
            q_idx
            if self._absolute_to_relative_idx is None
            else [self._absolute_to_relative_idx[idx] for idx in q_idx]
        )
        if self._delta_column_cache is not None:
            cached = self._delta_column_cache.get(key, relative_indices)
            if cached is not None:
                result[key] = cached
                continue
        try:
            result[key] = torch.stack(self.hf_dataset[key][relative_indices])
        except (KeyError, TypeError, IndexError):
            result[key] = torch.stack(self.hf_dataset[relative_indices][key])
    return result
```

- [ ] **Step 4: Run delta tests and verify GREEN**

Run:

```bash
uv run pytest tests/datasets/test_dataset_read_cache.py::test_delta_column_cache_serves_numeric_delta_queries tests/datasets/test_dataset_read_cache.py::test_delta_column_cache_preserves_episode_filter_indices -q
```

Expected: PASS.

- [ ] **Step 5: Commit delta cache**

Run:

```bash
git add src/lerobot/datasets/cache_utils.py src/lerobot/datasets/dataset_reader.py tests/datasets/test_dataset_read_cache.py
git commit -m "feat: cache delta timestamp numeric columns"
```

## Task 5: Regression Verification and Cleanup

**Files:**
- Modify if needed: `src/lerobot/datasets/cache_utils.py`
- Modify if needed: `src/lerobot/datasets/dataset_reader.py`
- Test: dataset reader and delta regression tests

- [ ] **Step 1: Run focused cache tests**

Run:

```bash
uv run pytest tests/datasets/test_dataset_read_cache.py -q
```

Expected: PASS.

- [ ] **Step 2: Run existing DatasetReader tests**

Run:

```bash
uv run pytest tests/datasets/test_dataset_reader.py -q
```

Expected: PASS.

- [ ] **Step 3: Run delta regression tests**

Run:

```bash
uv run pytest tests/datasets/test_datasets.py::test_delta_timestamps_query_returns_correct_values tests/datasets/test_datasets.py::test_delta_timestamps_with_episodes_filter tests/datasets/test_datasets.py::test_delta_timestamps_padding_at_episode_boundaries tests/datasets/test_datasets.py::test_delta_timestamps_multiple_episodes_filter -q
```

Expected: PASS.

- [ ] **Step 4: Inspect diff**

Run:

```bash
git diff -- src/lerobot/datasets/cache_utils.py src/lerobot/datasets/dataset_reader.py tests/datasets/test_dataset_read_cache.py
```

Expected: Diff is limited to dataset read cache behavior and tests.

- [ ] **Step 5: Commit cleanup if needed**

If any cleanup changes were made after Task 4, run:

```bash
git add src/lerobot/datasets/cache_utils.py src/lerobot/datasets/dataset_reader.py tests/datasets/test_dataset_read_cache.py
git commit -m "chore: polish dataset read cache"
```

## Self-Review Checklist

- The spec requirement for `root/image_cache/` is covered by Task 1 and Task 2.
- The spec requirement for `uint8` CHW memmap storage is covered by `ImageMemmapCache._expected_shape()` and `image_to_uint8_chw()`.
- The spec requirement for `index_hash` validation is covered by Task 3.
- The spec requirement for automatic first-access build is covered by `ImageMemmapCache.get()`.
- The spec requirement for non-visual numeric delta columns is covered by `DeltaColumnCache`.
- The spec requirement for episode filters is covered by delta episode-filter tests and index hash row-view hashing.
- The spec requirement that video keys remain on the existing path is covered by video exclusion test.
