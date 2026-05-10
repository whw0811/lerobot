from pathlib import Path

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")


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
