# Dataset Read Cache Design

## Goal

Add read-side caches to `LeRobotDataset` so repeated training reads avoid repeated Hugging Face Dataset image decoding and repeated delta-window numeric column queries.

The feature has two parts:

- Image cache: cache image-backed visual columns as `np.memmap` files in `root/image_cache/`.
- Delta column cache: cache non-visual numeric columns used by `delta_timestamps` as in-memory `torch.Tensor` objects.

## Scope

The first implementation only caches HF Dataset image columns where the feature dtype is `"image"`. Video-backed visual columns where dtype is `"video"` stay on the existing video decoding path.

The feature is read-only. It does not modify dataset parquet files, metadata, stats, Hub contents, recording behavior, or video files. It is active for normal `LeRobotDataset` reads and should work with optional episode filtering.

## Architecture

Create `src/lerobot/datasets/cache_utils.py` for cache-specific logic. `DatasetReader` remains responsible for read orchestration and delegates cache details to this module.

The module exposes two focused classes:

- `ImageMemmapCache`: manages image cache files, manifest files, index hashing, cache rebuilds, and image reads.
- `DeltaColumnCache`: stores selected numeric columns from the loaded HF Dataset as tensors and serves relative-index queries.

`DatasetReader` owns one optional instance of each class after the HF Dataset has loaded. It initializes them after `_build_index_mapping()` in both `try_load()` and `load_and_activate()`.

## Image Cache

Image cache files live under:

```text
root/image_cache/
```

Each image key gets its own sanitized filename:

```text
root/image_cache/<safe_key>.uint8_chw.memmap
root/image_cache/<safe_key>.manifest.json
```

The memmap stores frames as `uint8` in CHW layout:

```text
[num_frames, channels, height, width]
```

The manifest stores enough information to validate the file before reuse:

- `version`: integer cache format version.
- `key`: original feature key.
- `filename`: memmap filename.
- `dtype`: `"uint8"`.
- `layout`: `"CHW"`.
- `shape`: full memmap shape.
- `feature_shape`: shape from dataset metadata.
- `num_frames`: number of rows in the loaded HF Dataset view.
- `index_hash`: hash of the loaded HF Dataset `index` column.

The cache is built lazily on first access to an image key. If the manifest or memmap is missing, malformed, has a different shape, or has a different `index_hash`, the cache for that key is rebuilt.

During a rebuild, the implementation reads the currently loaded HF Dataset view in order and converts each image to uint8 CHW:

- Existing transformed HF image tensors are expected to be float CHW in `[0, 1]`; they are converted with clamp, multiply by 255, round, and cast to uint8.
- PIL or HWC array values are converted to CHW uint8 if encountered.

Reads from the cache return tensors compatible with existing behavior:

- `return_uint8=False`: return `torch.float32` CHW in `[0, 1]`.
- `return_uint8=True`: return `torch.uint8` CHW.

Image transforms still run after cached image reads, matching the existing `DatasetReader.get_item()` order.

## Index Hash

The index hash validates that the cached memmap corresponds to the loaded HF Dataset view, including episode filters.

The hash is computed from the ordered `index` column values and the row count. A stable SHA-256 digest is sufficient. The implementation should read the Arrow column directly when possible:

```python
indices = hf_dataset.data.column("index").to_numpy()
```

If that fast path is unavailable, it can fall back to `hf_dataset["index"]`.

When `episodes=None`, the hash covers the whole dataset. When `episodes=[...]`, the hash covers only the filtered rows. This makes cache reuse correct for each loaded view and avoids absolute/relative index mixups.

## Delta Column Cache

`DeltaColumnCache` is an in-memory cache built after the HF Dataset has loaded.

It only caches keys that satisfy all conditions:

- The key appears in `delta_indices`.
- The key is not in `meta.camera_keys`.
- The key exists in the loaded HF Dataset.
- The feature dtype is numeric and not `"image"` or `"video"`.

Examples include `action`, `observation.state`, and other state-like numeric tensors.

For each eligible key, the cache stores a `torch.Tensor` with first dimension equal to the loaded HF Dataset row count. Querying with relative indices returns:

```python
cached_tensor[relative_indices]
```

`DatasetReader._query_hf_dataset()` continues to accept absolute query indices. It converts to relative indices using `_absolute_to_relative_idx` when needed, then asks `DeltaColumnCache` first. If the cache has no entry for a key, it falls back to the current HF Dataset query logic.

This preserves behavior for non-numeric fields, uncached keys, and future feature types.

## Data Flow

Read initialization:

1. `DatasetReader.try_load()` or `DatasetReader.load_and_activate()` loads the HF Dataset.
2. `_build_index_mapping()` builds absolute-to-relative mapping for episode-filtered datasets.
3. `_init_read_caches()` constructs `DeltaColumnCache` and `ImageMemmapCache`.
4. Delta numeric columns are preloaded into memory immediately.
5. Image memmaps are not built until an image key is first requested.

Item read:

1. `get_item(idx)` reads the current HF Dataset row as before.
2. The item supplies `episode_index`, absolute `index`, and scalar fields.
3. If the item contains image-backed keys, `ImageMemmapCache` replaces those values with cached tensors.
4. If `delta_indices` exists, `_query_hf_dataset()` computes relative indices and reads numeric windows from `DeltaColumnCache` when available.
5. Video timestamps and video decoding stay unchanged.
6. Image transforms run after cache reads.
7. Task and subtask strings are added as before.

## Error Handling

Image cache rebuild is local to each image key. A stale or invalid cache for one key does not invalidate other keys.

If cache construction fails due to unsupported image shape or dtype, the error should identify the feature key and row index. Silent fallback would hide corrupt cache or data problems and make training nondeterministic.

If writing a memmap fails, the partial file and manifest should not be considered valid on the next run. The implementation should write the manifest only after the memmap has been fully flushed.

Delta cache construction should be conservative. If a requested delta key cannot be materialized as a tensor, it should skip caching that key and rely on the existing HF Dataset fallback.

## Testing

Add focused tests under `tests/datasets/test_dataset_reader.py` or a new `tests/datasets/test_dataset_read_cache.py`.

Required behavior:

- First access to an image-backed dataset creates `root/image_cache/`, a memmap, and a manifest.
- A second dataset instance with the same root and loaded index hash reuses the existing memmap.
- A changed or mismatched `index_hash` causes the image cache for that key to rebuild.
- Cached image reads match existing dataset image values and preserve dtype behavior for `return_uint8=False` and `return_uint8=True`.
- Delta queries for numeric non-visual keys return the same values as the existing HF Dataset path.
- Delta cache works with `episodes=[...]`, preserving absolute-to-relative index handling at episode boundaries.
- Video-backed datasets still read through the existing video path and do not create image memmaps for video keys.

Targeted verification commands:

```bash
uv run pytest tests/datasets/test_dataset_reader.py -q
uv run pytest tests/datasets/test_datasets.py::test_delta_timestamps_query_returns_correct_values -q
uv run pytest tests/datasets/test_datasets.py::test_delta_timestamps_with_episodes_filter -q
```

If the new tests live in a separate file, run that file directly as well.

## Compatibility Notes

Existing public constructor arguments remain unchanged. The cache is automatic and internal.

Existing returned item shapes and dtypes remain unchanged for default reads. Image cache stores uint8 on disk for compactness, but default reads still return float CHW images in `[0, 1]`, preserving the current HF transform behavior.

The cache is per local dataset root. It is not pushed to the Hub and should not be treated as source data.
