import json
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

    manifest = json.loads(manifest_path.read_text())
    manifest["index_hash"] = "stale"
    manifest_path.write_text(json.dumps(manifest))

    reloaded = LeRobotDataset(dataset.repo_id, root=dataset.root, download_videos=False)
    _ = reloaded[0]

    assert json.loads(manifest_path.read_text())["index_hash"] != "stale"


def test_video_backed_dataset_does_not_create_image_memmap(tmp_path, lerobot_dataset_factory):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=6,
        use_videos=True,
    )

    assert len(dataset.meta.image_keys) == 0
    assert not (dataset.root / "image_cache").exists()


def test_delta_column_cache_serves_numeric_delta_queries(
    tmp_path, empty_lerobot_dataset_factory, monkeypatch
):
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
    assert item["observation.state"].tolist() == [1.0, 2.0]
    assert item["action"].tolist() == [102.0, 103.0]


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

    assert item["observation.state"].tolist() == [11.0, 12.0]
    assert item["observation.state_is_pad"].tolist() == [False, False]
