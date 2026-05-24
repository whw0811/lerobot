from __future__ import annotations

import argparse
import bisect
import shutil
import traceback
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset


def progress_iter(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


class LambdaHead(nn.Module):
    default_head_hidden_size = 1024

    def __init__(
        self,
        hidden_size: int,
        head_hidden_size: int | None = None,
        dropout: float = 0.1,
        num_bins: int = 51,
    ):
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        head_hidden_size = head_hidden_size or self.default_head_hidden_size
        self.num_bins = num_bins
        self.head_hidden_size = head_hidden_size

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, num_bins),
        )

    def forward(self, pooled_hidden: Tensor) -> Tensor:
        return self.net(pooled_hidden.float())


def lambda_bin_values(num_bins: int, device: torch.device | str, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.linspace(0.0, 1.0, num_bins, device=device, dtype=dtype)


def lambda_expected_value(logits: Tensor) -> Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    bins = lambda_bin_values(logits.shape[-1], device=logits.device, dtype=probs.dtype)
    return (probs * bins).sum(dim=-1)


def lambda_target_bins(lambda_t: Tensor, num_bins: int, device: torch.device | str) -> Tensor:
    lambda_t = lambda_t.to(device=device, dtype=torch.float32).clamp(0.0, 1.0).view(-1)
    return torch.round(lambda_t * (num_bins - 1)).to(dtype=torch.long)


def compute_lambda_distribution_loss(
    lambda_logits: Tensor,
    lambda_t: Tensor,
    valid: Tensor,
    distance_weight: float = 0.0,
    distance_power: float = 1.0,
) -> Tensor:
    valid = valid.to(device=lambda_logits.device, dtype=torch.bool).view(-1)
    if not bool(valid.any().item()):
        return lambda_logits.sum() * 0.0
    target_bins = lambda_target_bins(lambda_t, lambda_logits.shape[-1], lambda_logits.device)
    loss = torch.nn.functional.cross_entropy(lambda_logits[valid], target_bins[valid])
    if distance_weight <= 0:
        return loss

    probs = torch.softmax(lambda_logits.float(), dim=-1)
    bins = lambda_bin_values(lambda_logits.shape[-1], device=lambda_logits.device, dtype=probs.dtype)
    target_values = lambda_t.to(device=lambda_logits.device, dtype=probs.dtype).clamp(0.0, 1.0).view(-1, 1)
    distances = (bins.view(1, -1) - target_values).abs().pow(distance_power)
    distance_loss = (probs * distances).sum(dim=-1)[valid].mean()
    return loss + distance_weight * distance_loss


def compute_lambda_tolerance_loss(
    lambda_hat: Tensor,
    lambda_t: Tensor,
    valid: Tensor,
    tolerance_margin: float,
    loss_type: str,
) -> Tensor:
    valid = valid.to(device=lambda_hat.device, dtype=torch.bool).view_as(lambda_hat)
    if not bool(valid.any().item()):
        return lambda_hat.sum() * 0.0

    lambda_t = lambda_t.to(device=lambda_hat.device, dtype=lambda_hat.dtype).view_as(lambda_hat)
    error = (lambda_hat - lambda_t).abs()
    excess = (error - tolerance_margin).clamp_min(0.0)
    excess = excess[valid]
    target = torch.zeros_like(excess)
    if loss_type == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(excess, target)
    if loss_type == "mse":
        return torch.nn.functional.mse_loss(excess, target)
    raise ValueError("loss_type must be 'smooth_l1' or 'mse'")


def resolve_cache_storage_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        return None
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported cache storage dtype: {dtype_name}")


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        usage = shutil.disk_usage(path.parent)
        free_gb = usage.free / 1e9
        raise RuntimeError(
            f"Failed to write cache file {path}. Free space on {path.parent}: {free_gb:.2f} GB. "
            "This usually means the disk/quota is full or the filesystem rejected the write."
        ) from exc


def resolve_shard_cache_dir(cache_path: str | Path) -> Path:
    cache_path = Path(cache_path)
    if cache_path.is_dir():
        return cache_path
    return Path(f"{cache_path}.shards")


class PrefixHiddenShardDataset(Dataset):
    def __init__(self, shard_dir: str | Path):
        self.shard_dir = Path(shard_dir)
        meta_path = self.shard_dir / "meta.pt"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing shard cache metadata: {meta_path}")
        self.meta = torch.load(meta_path, map_location="cpu")
        self.shards = self.meta["shards"]
        self.sizes = [int(shard["num_samples"]) for shard in self.shards]
        self.cumulative_sizes: list[int] = []
        running_total = 0
        for size in self.sizes:
            running_total += size
            self.cumulative_sizes.append(running_total)
        self.hidden_size = int(self.meta["hidden_size"])
        self._cached_shard_idx: int | None = None
        self._cached_shard: dict[str, Tensor] | None = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _load_shard(self, shard_idx: int) -> dict[str, Tensor]:
        if self._cached_shard_idx != shard_idx or self._cached_shard is None:
            shard_path = self.shard_dir / self.shards[shard_idx]["filename"]
            self._cached_shard = torch.load(shard_path, map_location="cpu")
            self._cached_shard_idx = shard_idx
        return self._cached_shard

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self.cumulative_sizes, index)
        previous_end = self.cumulative_sizes[shard_idx - 1] if shard_idx > 0 else 0
        local_idx = index - previous_end
        shard = self._load_shard(shard_idx)
        if "attention_mask" not in shard:
            return (
                shard["hidden_states"][local_idx],
                shard["indices"][local_idx],
            )
        return (
            shard["hidden_states"][local_idx],
            shard["attention_mask"][local_idx],
            shard["indices"][local_idx],
        )


def scalar_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def lambda_float(value: Any | None) -> float:
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        tensor = value.detach().float().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.as_tensor(np.asarray(value, dtype=np.float32))
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        return float(tensor.clamp(0.0, 1.0).item())
    if tensor.numel() == 1:
        return float(tensor.reshape(()).clamp(0.0, 1.0).item())
    return float(tensor.reshape(-1)[0].clamp(0.0, 1.0).item())


def to_hwc_uint8(image: Any) -> np.ndarray:
    array = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {array.shape}")
    if array.shape[0] in {1, 3, 4} and (
        array.shape[-1] not in {1, 3, 4} or array.shape[1] not in {1, 3, 4}
    ):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    else:
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def stack_view_images(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("At least one image is required")
    if any(image.ndim != 3 for image in images):
        raise ValueError("All view images must be HWC arrays")
    max_height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if image.shape[0] == max_height:
            padded.append(image)
            continue
        pad_height = max_height - image.shape[0]
        padding = np.zeros((pad_height, image.shape[1], image.shape[2]), dtype=image.dtype)
        padded.append(np.concatenate([image, padding], axis=0))
    return np.concatenate(padded, axis=1)


def overlay_text(image: np.ndarray, text: str) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install OpenCV with: pip install opencv-python") from exc

    annotated = image.copy()
    height, width = annotated.shape[:2]
    font_scale = 0.38
    thickness = 1
    padding = 5
    parts = text.split()
    lambda_parts = [part for part in parts if part.startswith(("lambda=", "lambda_hat="))]
    metadata = " ".join(part for part in parts if not part.startswith(("lambda=", "lambda_hat=")))
    lines = [metadata, *lambda_parts] if lambda_parts else [text]
    line_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness) for line in lines]
    line_height = max(size[0][1] + size[1] for size in line_sizes)
    box_width = min(width, max(size[0][0] for size in line_sizes) + 2 * padding)
    box_height = min(height, line_height * len(lines) + 2 * padding)
    background = annotated.copy()
    cv2.rectangle(background, (0, 0), (box_width, box_height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(background, 0.55, annotated, 0.45, 0.0, annotated)
    for line_idx, line in enumerate(lines):
        text_size, _baseline = line_sizes[line_idx]
        y = padding + text_size[1] + line_height * line_idx
        cv2.putText(
            annotated,
            line,
            (padding, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def format_lambda_debug_line(
    episode: int,
    frame: int,
    index: int,
    lambda_value: Any,
    lambda_hat: Any,
    valid: Any,
) -> str:
    valid_value = bool(valid.item()) if torch.is_tensor(valid) else bool(valid)
    return (
        f"ep={episode} frame={frame} index={index} "
        f"lambda={lambda_float(lambda_value):.6f} "
        f"lambda_hat={lambda_float(lambda_hat):.6f} "
        f"valid={valid_value}"
    )


def get_task_text(item: dict[str, Any]) -> str:
    task = item.get("task", "")
    if isinstance(task, (list, tuple)):
        task = task[0] if task else ""
    return str(task)


def build_task_prompt(task: str, num_images: int) -> str:
    image_tokens = "\n".join(["<image>"] * num_images)
    return f"{image_tokens}\nTask: {task}\nPredict the task difficulty lambda."


def choose_camera_keys(dataset: "LeRobotDataset", camera_keys: list[str] | None) -> list[str]:
    available = list(dataset.meta.camera_keys)
    if camera_keys:
        missing = [key for key in camera_keys if key not in available]
        if missing:
            raise ValueError(f"Camera keys {missing} are not in dataset cameras: {available}")
        return camera_keys[:2]
    if len(available) < 2:
        raise ValueError(f"Need at least two camera views, found: {available}")
    return available[:2]


def make_smolvla_debug_dataset(
    *,
    repo_id: str,
    root: str | Path | None,
    episodes: list[int] | None = None,
    chunk_size: int = 50,
    use_image_cache: bool = False,
    image_cache_dir: str | Path | None = None,
    build_image_cache: bool = True,
    revision: str | None = None,
    video_backend: str | None = None,
    tolerance_s: float = 1e-4,
    dataset_cls: Any | None = None,
    metadata_cls: Any | None = None,
) -> "LeRobotDataset":
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if dataset_cls is None:
        from lerobot.datasets import LeRobotDataset as dataset_cls
    if metadata_cls is None:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata as metadata_cls
    from lerobot.utils.constants import ACTION

    ds_meta = metadata_cls(repo_id, root=root, revision=revision)
    action_delta_timestamps = [idx / ds_meta.fps for idx in range(chunk_size)]
    return dataset_cls(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps={ACTION: action_delta_timestamps},
        revision=revision,
        video_backend=video_backend,
        return_uint8=True,
        use_image_cache=use_image_cache,
        image_cache_dir=image_cache_dir,
        build_image_cache=build_image_cache,
        tolerance_s=tolerance_s,
    )


def masked_mean_pool(hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
    hidden = hidden.float()
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def convert_uint8_images_to_float(batch: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    converted = dict(batch)
    for camera_key in camera_keys:
        value = converted.get(camera_key)
        if torch.is_tensor(value) and value.dtype == torch.uint8:
            converted[camera_key] = value.to(dtype=torch.float32) / 255.0
    return converted


def extract_smolvla_prefix_hidden(policy, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    with torch.inference_mode():
        batch = policy._prepare_batch(batch)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_outputs, _past_key_values = policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        prefix_hidden = prefix_outputs[0]
    return prefix_hidden.detach().clone(), prefix_pad_masks.detach().clone()


def pool_prefix_hidden(prefix_hidden: Tensor, prefix_pad_masks: Tensor) -> Tensor:
    return masked_mean_pool(prefix_hidden, prefix_pad_masks)


def get_smolvla_prefix_hidden(
    policy,
    preprocessor,
    batch: dict[str, Any],
    camera_keys: list[str],
) -> tuple[Tensor, Tensor]:
    batch = convert_uint8_images_to_float(batch, camera_keys)
    processed_batch = preprocessor(batch)
    return extract_smolvla_prefix_hidden(policy, processed_batch)


def compute_lambda_loss(
    lambda_logits: Tensor,
    lambda_t: Tensor,
    valid: Tensor,
    loss_type: str,
    variance_weight: float = 0.1,
    tolerance_margin: float = 0.0,
    tolerance_weight: float = 0.0,
    distance_weight: float = 0.0,
    distance_power: float = 1.0,
) -> Tensor:
    distribution_loss = compute_lambda_distribution_loss(
        lambda_logits,
        lambda_t,
        valid,
        distance_weight=distance_weight,
        distance_power=distance_power,
    )
    lambda_hat = lambda_expected_value(lambda_logits)
    lambda_t = lambda_t.to(device=lambda_hat.device, dtype=lambda_hat.dtype).view_as(lambda_hat)
    valid = valid.to(device=lambda_hat.device, dtype=torch.bool).view_as(lambda_hat)
    loss = distribution_loss
    if tolerance_weight > 0:
        tolerance_loss = compute_lambda_tolerance_loss(
            lambda_hat,
            lambda_t,
            valid,
            tolerance_margin=tolerance_margin,
            loss_type=loss_type,
        )
        loss = loss + tolerance_weight * tolerance_loss
    if variance_weight > 0 and bool(valid.sum().item() > 1):
        valid_hat = lambda_hat[valid]
        valid_t = lambda_t[valid]
        variance_loss = torch.nn.functional.mse_loss(valid_hat.var(), valid_t.var())
        loss = loss + variance_weight * variance_loss
    return loss


def render_lambda_debug_video(
    *,
    policy,
    preprocessor,
    head: LambdaHead,
    repo_id: str,
    root: str | Path,
    labels_path: str | Path,
    episode: int,
    output_path: str | Path,
    device: str,
    camera_keys: list[str] | None = None,
    max_frames: int | None = None,
    fps: float | None = None,
    default_value: float = 0.0,
    chunk_size: int = 50,
    use_image_cache: bool = False,
    image_cache_dir: str | Path | None = None,
    build_image_cache: bool = True,
    revision: str | None = None,
    video_backend: str | None = None,
    tolerance_s: float = 1e-4,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install OpenCV with: pip install opencv-python") from exc
    from lerobot.policies.smolvla.lambda_labels import LambdaLabelLookup, load_lambda_sidecar

    dataset = make_smolvla_debug_dataset(
        repo_id=repo_id,
        root=root,
        episodes=[episode],
        chunk_size=chunk_size,
        use_image_cache=use_image_cache,
        image_cache_dir=image_cache_dir,
        build_image_cache=build_image_cache,
        revision=revision,
        video_backend=video_backend,
        tolerance_s=tolerance_s,
    )
    selected_cameras = choose_camera_keys(dataset, camera_keys)
    lookup = LambdaLabelLookup(load_lambda_sidecar(labels_path), default_value=default_value)
    playback_fps = float(fps if fps is not None else dataset.fps)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    num_frames = dataset.num_frames if max_frames is None else min(dataset.num_frames, max_frames)

    policy.eval()
    head.eval()
    try:
        frame_iter = progress_iter(
            range(num_frames),
            desc=f"render ep{episode}",
            unit="frame",
            leave=False,
        )
        for frame in frame_iter:
            item = dataset[frame]
            index = scalar_int(item["index"])
            values, _confidence, valid = lookup.lookup_with_confidence(torch.tensor([index], dtype=torch.long))
            batch = {
                key: (value.unsqueeze(0) if torch.is_tensor(value) else [value])
                for key, value in item.items()
            }
            prefix_hidden, prefix_mask = get_smolvla_prefix_hidden(
                policy, preprocessor, batch, list(dataset.meta.camera_keys)
            )
            hidden = pool_prefix_hidden(prefix_hidden, prefix_mask)
            with torch.no_grad():
                lambda_logits = head(hidden)
                lambda_hat = lambda_expected_value(lambda_logits)
            line = format_lambda_debug_line(
                episode=episode,
                frame=frame,
                index=index,
                lambda_value=values[0],
                lambda_hat=lambda_hat,
                valid=valid[0],
            )
            views = [to_hwc_uint8(item[camera_key]) for camera_key in selected_cameras]
            display_image = overlay_text(stack_view_images(views), line)
            bgr_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
            if writer is None:
                height, width = bgr_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, playback_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {output_path}")
            writer.write(bgr_image)
    finally:
        if writer is not None:
            writer.release()
        head.train()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a standalone lambda head on frozen VLM hidden states.")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--root", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--labels-path", default=None)
    parser.add_argument("--vlm-model-name", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--vlm-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default=None,
        help="Deprecated for SmolVLA prefix extraction; SmolVLA loads the VLM using its policy defaults.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Action chunk size used to read action windows like SmolVLA training.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--default-value", type=float, default=0.0)
    parser.add_argument("--loss-type", choices=["smooth_l1", "mse"], default="smooth_l1")
    parser.add_argument(
        "--variance-weight",
        type=float,
        default=0.1,
        help="Weight for variance loss to prevent convergence to constant value",
    )
    parser.add_argument(
        "--tolerance-margin",
        type=float,
        default=0.0,
        help="Dead-zone margin around lambda labels where expected-value regression is not penalized.",
    )
    parser.add_argument(
        "--tolerance-weight",
        type=float,
        default=0.0,
        help="Weight for tolerance loss on the distribution expected value.",
    )
    parser.add_argument(
        "--distance-weight",
        type=float,
        default=0.0,
        help="Weight for expected bin distance inside the distribution loss.",
    )
    parser.add_argument(
        "--distance-power",
        type=float,
        default=1.0,
        help="Power applied to bin distance from the lambda label; 1 is linear, 2 emphasizes far bins.",
    )
    parser.add_argument("--head-hidden-size", type=int, default=None)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--lambda-num-bins", type=int, default=51, help="Number of discrete bins for distributional lambda prediction.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--test-episodes",
        type=str,
        default="1",
        help="Episodes to render after training (comma-separated, e.g., '1,2,3')",
    )
    parser.add_argument("--vis-max-frames", type=int, default=100)
    parser.add_argument("--vis-fps", type=float, default=10.0)
    parser.add_argument("--vis-output-dir", default="lambda_debug_videos")
    parser.add_argument("--camera-key", action="append", default=None, help="Camera key to use. Pass twice.")
    parser.add_argument("--save-head-path", default=None)
    parser.add_argument(
        "--cache-mode",
        choices=["none", "save", "load"],
        default="none",
        help="Cache mode: none (no caching), save (extract and save SmolVLA prefix hidden), load (load from cache)",
    )
    parser.add_argument("--cache-path", default="vlm_hidden_cache.pt", help="Path to save/load VLM hidden states cache")
    parser.add_argument("--cache-batch-size", type=int, default=32, help="Batch size for VLM feature extraction during caching")
    parser.add_argument(
        "--cache-shard-batches",
        type=int,
        default=1,
        help="Number of extraction batches to store in each shard cache file.",
    )
    parser.add_argument(
        "--cache-storage-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="float16",
        help="Dtype used to store prefix hidden shards. float16 reduces disk usage; auto keeps model output dtype.",
    )
    parser.add_argument(
        "--use-image-cache",
        action="store_true",
        help="Read image-backed observations through the dataset image cache.",
    )
    parser.add_argument(
        "--image-cache-dir",
        default=None,
        help="Directory for the dataset image cache. Defaults to root/image_cache.",
    )
    parser.add_argument(
        "--no-build-image-cache",
        action="store_true",
        help="Require an existing valid image cache instead of building one.",
    )
    return parser


def load_smolvla_policy_and_preprocessor(
    dataset,
    model_name: str,
    device: str,
    chunk_size: int,
):
    from lerobot.configs import FeatureType
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION
    from lerobot.utils.feature_utils import dataset_to_policy_features

    print("building SmolVLA policy for prefix hidden extraction", flush=True)
    cfg = SmolVLAConfig(
        vlm_model_name=model_name,
        load_vlm_weights=True,
        device=device,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        pad_language_to="max_length",
    )
    features = dataset_to_policy_features(dataset.meta.features)
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    if ACTION not in cfg.output_features:
        raise ValueError(f"Dataset features do not contain required action key '{ACTION}'")

    policy = SmolVLAPolicy(cfg).to(device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    preprocessor, _postprocessor = make_smolvla_pre_post_processors(cfg, dataset.meta.stats)
    return policy, preprocessor


def extract_and_cache_smolvla_prefix_hidden(
    dataset,
    policy,
    preprocessor,
    camera_keys,
    cache_path,
    batch_size=32,
    num_workers=0,
    shard_batches=1,
    storage_dtype: torch.dtype | None = torch.float16,
):
    print(
        f"Extracting pooled SmolVLA prefix hidden states for {len(dataset)} frames with batch_size={batch_size}...",
        flush=True,
    )
    if shard_batches <= 0:
        raise ValueError("shard_batches must be positive")

    cache_path = Path(cache_path)
    shard_dir = resolve_shard_cache_dir(cache_path)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in shard_dir.glob("shard_*.pt"):
        stale_path.unlink()
    meta_path = shard_dir / "meta.pt"
    if meta_path.exists():
        meta_path.unlink()

    hidden_states = []
    indices = []
    shard_meta = []
    shard_idx = 0
    hidden_size = None

    def flush_shard() -> None:
        nonlocal shard_idx, hidden_size
        if not hidden_states:
            return
        hidden_tensor = torch.cat(hidden_states, dim=0)
        if storage_dtype is not None:
            hidden_tensor = hidden_tensor.to(dtype=storage_dtype)
        indices_tensor = torch.cat([index.view(-1) for index in indices], dim=0)
        hidden_size = int(hidden_tensor.shape[-1])
        shard_filename = f"shard_{shard_idx:06d}.pt"
        shard_path = shard_dir / shard_filename
        atomic_torch_save(
            {
                "hidden_states": hidden_tensor,
                "indices": indices_tensor,
            },
            shard_path,
        )
        shard_bytes = shard_path.stat().st_size
        shard_meta.append(
            {
                "filename": shard_filename,
                "num_samples": int(hidden_tensor.shape[0]),
                "bytes": int(shard_bytes),
            }
        )
        print(
            f"Saved shard {shard_idx:06d}: samples={hidden_tensor.shape[0]} "
            f"size={shard_bytes / 1e9:.2f} GB",
            flush=True,
        )
        shard_idx += 1
        hidden_states.clear()
        indices.clear()
        del hidden_tensor, indices_tensor

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch_idx, batch in enumerate(progress_iter(loader, desc="extracting SmolVLA prefix", unit="batch")):
        prefix_hidden, prefix_mask = get_smolvla_prefix_hidden(policy, preprocessor, batch, camera_keys)
        pooled_hidden = pool_prefix_hidden(prefix_hidden, prefix_mask)
        hidden_states.append(pooled_hidden.detach().cpu())
        indices.append(torch.as_tensor(batch["index"], dtype=torch.long).cpu())
        del prefix_hidden, prefix_mask, pooled_hidden, batch
        if (batch_idx + 1) % shard_batches == 0:
            flush_shard()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    flush_shard()
    if hidden_size is None:
        raise RuntimeError("No prefix hidden states were extracted")

    meta = {
        "format": "smolvla_prefix_shards",
        "version": 1,
        "shards": shard_meta,
        "num_samples": sum(int(shard["num_samples"]) for shard in shard_meta),
        "hidden_size": hidden_size,
        "storage_dtype": str(storage_dtype).removeprefix("torch.") if storage_dtype is not None else "auto",
        "source": "smolvla_prefix_global_mean_pooling",
    }
    atomic_torch_save(meta, meta_path)
    print(
        f"Saved pooled SmolVLA prefix hidden shard cache to {shard_dir} "
        f"({meta['num_samples']} samples, {len(shard_meta)} shards)",
        flush=True,
    )

    return PrefixHiddenShardDataset(shard_dir)


def load_cached_hidden(cache_path):
    shard_dir = resolve_shard_cache_dir(cache_path)
    if shard_dir.exists():
        dataset = PrefixHiddenShardDataset(shard_dir)
        print(
            f"Loaded shard cache from {shard_dir}: samples={len(dataset)}, hidden_size={dataset.hidden_size}, "
            f"shards={len(dataset.shards)}",
            flush=True,
        )
        return dataset

    print(f"Loading VLM hidden states from {cache_path}...", flush=True)
    cache_data = torch.load(cache_path)
    print(f"Loaded cache: hidden_states shape={cache_data['hidden_states'].shape}, indices shape={cache_data['indices'].shape}", flush=True)
    return cache_data


def main() -> None:
    args = build_arg_parser().parse_args()

    print(f"loading dataset from {args.root}", flush=True)
    dataset = make_smolvla_debug_dataset(
        repo_id=args.repo_id,
        root=args.root,
        chunk_size=args.chunk_size,
        use_image_cache=args.use_image_cache,
        image_cache_dir=args.image_cache_dir,
        build_image_cache=not args.no_build_image_cache,
        revision=args.revision,
        video_backend=args.video_backend,
        tolerance_s=args.tolerance_s,
    )
    camera_keys = choose_camera_keys(dataset, args.camera_key)
    print(
        f"dataset frames={dataset.num_frames} episodes={dataset.num_episodes} cameras={camera_keys}",
        flush=True,
    )

    policy = None
    preprocessor = None
    cache_data = None
    
    if args.cache_mode == "load":
        cache_data = load_cached_hidden(args.cache_path)
        hidden_size = cache_data.hidden_size if isinstance(cache_data, PrefixHiddenShardDataset) else cache_data["hidden_size"]
    else:
        print(f"loading SmolVLA prefix model from {args.vlm_model_name} on {args.device}", flush=True)
        policy, preprocessor = load_smolvla_policy_and_preprocessor(
            dataset=dataset,
            model_name=args.vlm_model_name,
            device=args.device,
            chunk_size=args.chunk_size,
        )

        print("probing SmolVLA prefix hidden size", flush=True)
        first_batch = {
            key: (value.unsqueeze(0) if torch.is_tensor(value) else [value])
            for key, value in dataset[0].items()
        }
        prefix_hidden, _prefix_mask = get_smolvla_prefix_hidden(
            policy, preprocessor, first_batch, list(dataset.meta.camera_keys)
        )
        hidden_size = prefix_hidden.shape[-1]

        if args.cache_mode == "save":
            cache_data = extract_and_cache_smolvla_prefix_hidden(
                dataset,
                policy,
                preprocessor,
                list(dataset.meta.camera_keys),
                args.cache_path,
                args.cache_batch_size,
                args.num_workers,
                args.cache_shard_batches,
                resolve_cache_storage_dtype(args.cache_storage_dtype),
            )
            print("Pooled SmolVLA prefix hidden states cached, you can now use --cache-mode=load for faster training", flush=True)
            return

    if args.labels_path is None:
        raise ValueError("--labels-path is required unless --cache-mode=save")

    from lerobot.policies.smolvla.lambda_labels import LambdaLabelLookup, load_lambda_sidecar

    print(f"loading lambda labels from {args.labels_path}", flush=True)
    lookup = LambdaLabelLookup(load_lambda_sidecar(args.labels_path), default_value=args.default_value)

    print(f"hidden_size={hidden_size}; training lambda head for {args.steps} steps", flush=True)
    head = LambdaHead(
        hidden_size,
        head_hidden_size=args.head_hidden_size,
        dropout=args.head_dropout,
        num_bins=args.lambda_num_bins,
    ).to(args.device)
    
    trainable_parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    
    if cache_data is not None:
        from torch.utils.data import TensorDataset
        if isinstance(cache_data, PrefixHiddenShardDataset):
            hidden_dataset = cache_data
        elif "attention_mask" in cache_data:
            hidden_dataset = TensorDataset(
                cache_data["hidden_states"], cache_data["attention_mask"], cache_data["indices"]
            )
        else:
            hidden_dataset = TensorDataset(cache_data["hidden_states"], cache_data["indices"])
        loader = DataLoader(hidden_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        iterator = cycle(loader)
        use_cache = True
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        iterator = cycle(loader)
        use_cache = False

    train_iter = progress_iter(range(args.steps), desc="lambda-head training", unit="step")
    for step in train_iter:
        batch = next(iterator)
        
        if use_cache:
            if len(batch) == 3:
                hidden_batch, mask_batch, indices_batch = batch
                prefix_hidden = hidden_batch.to(args.device)
                prefix_mask = mask_batch.to(args.device)
                hidden = pool_prefix_hidden(prefix_hidden, prefix_mask)
            else:
                hidden_batch, indices_batch = batch
                hidden = hidden_batch.to(args.device)
            indices = indices_batch
        else:
            indices = torch.as_tensor(batch["index"], dtype=torch.long)
            prefix_hidden, prefix_mask = get_smolvla_prefix_hidden(
                policy, preprocessor, batch, list(dataset.meta.camera_keys)
            )
            hidden = pool_prefix_hidden(prefix_hidden, prefix_mask)
        
        lambda_t, _confidence, valid = lookup.lookup_with_confidence(indices)
        lambda_logits = head(hidden)
        lambda_hat = lambda_expected_value(lambda_logits)
        loss = compute_lambda_loss(
            lambda_logits,
            lambda_t,
            valid,
            args.loss_type,
            args.variance_weight,
            args.tolerance_margin,
            args.tolerance_weight,
            args.distance_weight,
            args.distance_power,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_values = {
            "loss": float(loss.item()),
            "hat_mean": float(lambda_hat.detach().mean().item()),
        }
        if hasattr(train_iter, "set_postfix"):
            train_iter.set_postfix(log_values)

        if args.log_every > 0 and step % args.log_every == 0:
            valid_mask = valid.to(device=lambda_hat.device, dtype=torch.bool).view_as(lambda_hat)
            lambda_t_device = lambda_t.to(device=lambda_hat.device)
            valid_mean = (
                lambda_hat.detach()[valid_mask].float().mean().item()
                if bool(valid_mask.any().item())
                else 0.0
            )
            valid_hat = lambda_hat.detach()[valid_mask]
            valid_t = lambda_t_device[valid_mask]
            hat_var = valid_hat.var().item() if bool(valid_mask.any().item()) else 0.0
            t_mean = valid_t.float().mean().item() if bool(valid_mask.any().item()) else 0.0
            t_var = valid_t.float().var().item() if bool(valid_mask.any().item()) else 0.0
            print(
                f"step={step}/{args.steps} "
                f"loss={log_values['loss']:.6f} "
                f"lambda_hat: mean={log_values['hat_mean']:.4f}, var={hat_var:.4f}, min={valid_hat.min().item() if bool(valid_mask.any().item()) else 0:.4f}, max={valid_hat.max().item() if bool(valid_mask.any().item()) else 0:.4f} "
                f"lambda_t: mean={t_mean:.4f}, var={t_var:.4f}, min={valid_t.min().item() if bool(valid_mask.any().item()) else 0:.4f}, max={valid_t.max().item() if bool(valid_mask.any().item()) else 0:.4f}",
                flush=True,
            )
            if bool(valid_mask.any().item()):
                hat_list = [f"{x:.4f}" for x in valid_hat.cpu().tolist()]
                t_list = [f"{x:.4f}" for x in valid_t.cpu().tolist()]
                print(f"  lambda_hat: [{', '.join(hat_list)}]", flush=True)
                print(f"  lambda_t:   [{', '.join(t_list)}]", flush=True)

    if args.save_head_path is not None:
        save_path = Path(args.save_head_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "hidden_size": hidden_size,
                "head_hidden_size": head.head_hidden_size,
                "head_dropout": args.head_dropout,
                "lambda_num_bins": args.lambda_num_bins,
                "head_architecture": "pi06_style_two_layer_mlp",
                "tolerance_margin": args.tolerance_margin,
                "tolerance_weight": args.tolerance_weight,
                "distance_weight": args.distance_weight,
                "distance_power": args.distance_power,
                "variance_weight": args.variance_weight,
                "camera_keys": camera_keys,
                "vlm_model_name": args.vlm_model_name,
            },
            save_path,
        )
        print(f"saved lambda head to {save_path}", flush=True)

    test_episodes = [int(ep.strip()) for ep in args.test_episodes.split(",")]
    print(f"\n=== Testing and rendering videos for episodes {test_episodes} ===", flush=True)
    if policy is None or preprocessor is None:
        print(f"loading SmolVLA prefix model from {args.vlm_model_name} on {args.device}", flush=True)
        policy, preprocessor = load_smolvla_policy_and_preprocessor(
            dataset=dataset,
            model_name=args.vlm_model_name,
            device=args.device,
            chunk_size=args.chunk_size,
        )
    policy.eval()
    head.eval()
    output_dir = Path(args.vis_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode in test_episodes:
        output_path = output_dir / f"lambda_debug_episode_{episode:03d}.mp4"
        print(f"Rendering episode {episode} to {output_path}...", flush=True)
        render_lambda_debug_video(
            policy=policy,
            preprocessor=preprocessor,
            head=head,
            repo_id=args.repo_id,
            root=args.root,
            labels_path=args.labels_path,
            episode=episode,
            output_path=output_path,
            device=args.device,
            camera_keys=camera_keys,
            max_frames=args.vis_max_frames,
            fps=args.vis_fps,
            default_value=args.default_value,
            chunk_size=args.chunk_size,
            use_image_cache=args.use_image_cache,
            image_cache_dir=args.image_cache_dir,
            build_image_cache=not args.no_build_image_cache,
            revision=args.revision,
            video_backend=args.video_backend,
            tolerance_s=args.tolerance_s,
        )
        print(f"Saved {output_path}", flush=True)
    print("=== Testing complete ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
