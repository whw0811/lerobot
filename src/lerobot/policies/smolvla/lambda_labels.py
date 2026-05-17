#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from lerobot.configs.types import NormalizationMode
from lerobot.utils.constants import ACTION

DEFAULT_LAMBDA_CHUNK_SIZE = 10
DEFAULT_LAMBDA_ANCHOR_INDICES = (0, 3, 6, 9)
DEFAULT_LAMBDA_QUERY_INDICES = tuple(
    idx for idx in range(DEFAULT_LAMBDA_CHUNK_SIZE) if idx not in DEFAULT_LAMBDA_ANCHOR_INDICES
)


@dataclass(frozen=True)
class LambdaLabelConfig:
    chunk_size: int = DEFAULT_LAMBDA_CHUNK_SIZE
    anchor_indices: tuple[int, ...] = DEFAULT_LAMBDA_ANCHOR_INDICES
    query_indices: tuple[int, ...] = DEFAULT_LAMBDA_QUERY_INDICES
    position_weights: tuple[float, ...] | None = None
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        anchor_set = set(self.anchor_indices)
        query_set = set(self.query_indices)
        if anchor_set & query_set:
            raise ValueError("anchor_indices and query_indices must not overlap")
        all_indices = anchor_set | query_set
        if any(idx < 0 or idx >= self.chunk_size for idx in all_indices):
            raise ValueError("anchor_indices and query_indices must be inside chunk_size")
        if len(anchor_set) != len(self.anchor_indices) or len(query_set) != len(self.query_indices):
            raise ValueError("anchor_indices and query_indices must not contain duplicates")
        expected = set(range(self.chunk_size))
        if all_indices != expected:
            raise ValueError("anchor_indices and query_indices must cover every chunk position")
        if self.position_weights is not None:
            if len(self.position_weights) != self.chunk_size:
                raise ValueError("position_weights must have one value per chunk position")
            if any(weight < 0 for weight in self.position_weights):
                raise ValueError("position_weights must be non-negative")


class LambdaMetrics(NamedTuple):
    pchip_error: Tensor
    lambda_raw: Tensor
    lambda_envelope: Tensor
    lambda_t: Tensor
    q_low_value: Tensor
    q_high_value: Tensor


class ActionChunkBatch(NamedTuple):
    indices: Tensor
    episode_indices: Tensor
    chunks: Tensor


class PreloadedActionColumns(NamedTuple):
    indices: Tensor
    episode_indices: Tensor
    actions: Tensor
    row_by_index: dict[int, int]


class FramewisePchipErrors(NamedTuple):
    indices: Tensor
    episode_indices: Tensor
    pchip_error: Tensor


def compute_pchip_trend(action_chunks: Tensor, cfg: LambdaLabelConfig) -> Tensor:
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError as exc:
        raise ImportError(
            "PCHIP trend generation requires scipy. Install a SciPy-enabled extra such as "
            "`lerobot[scipy-dep]`, `lerobot[libero]`, or `lerobot[pi]`."
        ) from exc

    input_was_unbatched = action_chunks.ndim == 2
    if input_was_unbatched:
        action_chunks = action_chunks.unsqueeze(0)
    if action_chunks.ndim != 3:
        raise ValueError(f"Expected action chunks shaped [N, {cfg.chunk_size}, D] or [{cfg.chunk_size}, D]")
    if action_chunks.shape[1] != cfg.chunk_size:
        raise ValueError(f"Expected chunk_size={cfg.chunk_size}, got {action_chunks.shape[1]}")

    device = action_chunks.device
    dtype = action_chunks.dtype
    chunks_np = action_chunks.detach().cpu().float().numpy()
    x_anchor = list(cfg.anchor_indices)
    x_full = list(range(cfg.chunk_size))
    trend = torch.empty_like(action_chunks, device="cpu", dtype=torch.float32)

    for batch_idx in range(chunks_np.shape[0]):
        interpolator = PchipInterpolator(x_anchor, chunks_np[batch_idx, x_anchor, :], axis=0)
        trend[batch_idx] = torch.from_numpy(interpolator(x_full)).to(dtype=torch.float32)

    trend = trend.to(device=device, dtype=dtype)
    return trend.squeeze(0) if input_was_unbatched else trend


def _validate_quantile_range(q_low: float, q_high: float) -> None:
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("q_low and q_high must satisfy 0 <= q_low < q_high <= 1")


def _normalize_by_quantiles(values: Tensor, q_low: float, q_high: float, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    _validate_quantile_range(q_low, q_high)
    values_f32 = values.float()
    q_low_value = torch.quantile(values_f32, q_low)
    q_high_value = torch.quantile(values_f32, q_high)
    denom = torch.clamp(q_high_value - q_low_value, min=eps)
    normalized = torch.clamp((values_f32 - q_low_value) / denom, min=0.0, max=1.0)
    return normalized.to(dtype=values.dtype), q_low_value, q_high_value


def compute_dct_high_frequency_ratio(
    action_chunks: Tensor,
    high_freq_start: int | None = None,
    eps: float = 1e-6,
) -> Tensor:
    if action_chunks.ndim != 3:
        raise ValueError("Expected action_chunks shaped [N, T, D]")
    chunk_size = action_chunks.shape[1]
    if chunk_size < 2:
        raise ValueError("DCT high-frequency ratio requires chunk_size >= 2")
    if high_freq_start is None:
        high_freq_start = max(1, chunk_size // 2)
    if not 1 <= high_freq_start < chunk_size:
        raise ValueError("high_freq_start must be in [1, chunk_size)")

    chunks = action_chunks.float()
    centered = chunks - chunks.mean(dim=1, keepdim=True)
    n = torch.arange(chunk_size, dtype=chunks.dtype, device=chunks.device)
    k = torch.arange(chunk_size, dtype=chunks.dtype, device=chunks.device)
    basis = torch.cos(torch.pi / chunk_size * (n[None, :] + 0.5) * k[:, None])
    coeffs = torch.einsum("kt,btd->bkd", basis, centered)

    total_energy = coeffs[:, 1:, :].pow(2).sum(dim=(1, 2))
    high_energy = coeffs[:, high_freq_start:, :].pow(2).sum(dim=(1, 2))
    ratio = high_energy / torch.clamp(total_energy, min=eps)
    return torch.where(total_energy > eps, ratio, torch.zeros_like(ratio)).to(dtype=action_chunks.dtype)


def compute_lambda_confidence(
    lambda_pchip: Tensor,
    lambda_dct: Tensor,
    alpha_conf_min: float = 0.15,
    alpha_conf_max: float = 1.0,
) -> Tensor:
    if not 0.0 <= alpha_conf_min <= alpha_conf_max <= 1.0:
        raise ValueError("alpha_conf_min and alpha_conf_max must satisfy 0 <= min <= max <= 1")
    lambda_pchip = torch.as_tensor(lambda_pchip, dtype=lambda_dct.dtype, device=lambda_dct.device)
    consistency = 1.0 - torch.clamp((lambda_pchip - lambda_dct).abs(), min=0.0, max=1.0)
    return torch.clamp(
        alpha_conf_min + (alpha_conf_max - alpha_conf_min) * consistency,
        min=alpha_conf_min,
        max=alpha_conf_max,
    )


def _position_weights_tensor(cfg: LambdaLabelConfig, device: torch.device, dtype: torch.dtype) -> Tensor:
    if cfg.position_weights is None:
        return torch.ones(cfg.chunk_size, dtype=dtype, device=device)
    return torch.tensor(cfg.position_weights, dtype=dtype, device=device)


def compute_framewise_pchip_errors(
    chunks: Tensor,
    indices: Tensor,
    episode_indices: Tensor,
    cfg: LambdaLabelConfig,
) -> FramewisePchipErrors:
    if chunks.ndim != 3 or chunks.shape[1] != cfg.chunk_size:
        raise ValueError(f"Expected chunks shaped [N, {cfg.chunk_size}, D]")

    indices = torch.as_tensor(indices, dtype=torch.long, device=chunks.device).view(-1)
    episode_indices = torch.as_tensor(episode_indices, dtype=torch.long, device=chunks.device).view(-1)
    if chunks.shape[0] != indices.numel() or chunks.shape[0] != episode_indices.numel():
        raise ValueError("chunks, indices, and episode_indices must have matching first dimensions")

    trend = compute_pchip_trend(chunks, cfg)
    query_positions = torch.tensor(cfg.query_indices, dtype=torch.long, device=chunks.device)
    position_weights = _position_weights_tensor(cfg, chunks.device, chunks.dtype)
    query_weights = position_weights[query_positions]
    keep_positions = query_weights > 0
    if not torch.any(keep_positions):
        raise ValueError("At least one non-anchor position must have a positive aggregation weight")

    query_positions = query_positions[keep_positions]
    query_weights = query_weights[keep_positions]
    position_errors = (chunks - trend).pow(2).mean(dim=2)

    contribution_indices = indices[:, None] + query_positions[None, :]
    contribution_episodes = episode_indices[:, None].expand_as(contribution_indices)
    contribution_errors = position_errors[:, query_positions] * query_weights[None, :]
    contribution_weights = query_weights[None, :].expand_as(contribution_errors)

    flat_indices = contribution_indices.reshape(-1)
    flat_episodes = contribution_episodes.reshape(-1)
    flat_errors = contribution_errors.reshape(-1)
    flat_weights = contribution_weights.reshape(-1)

    valid_indices, inverse = torch.unique(flat_indices, sorted=True, return_inverse=True)
    weighted_error_sum = torch.zeros(valid_indices.numel(), dtype=chunks.dtype, device=chunks.device)
    weight_sum = torch.zeros_like(weighted_error_sum)
    weighted_error_sum.scatter_add_(0, inverse, flat_errors)
    weight_sum.scatter_add_(0, inverse, flat_weights)
    pchip_error = weighted_error_sum / torch.clamp(weight_sum, min=cfg.eps)

    first_rows = torch.full((valid_indices.numel(),), flat_indices.numel(), dtype=torch.long, device=chunks.device)
    source_rows = torch.arange(flat_indices.numel(), dtype=torch.long, device=chunks.device)
    first_rows.scatter_reduce_(0, inverse, source_rows, reduce="amin", include_self=True)
    valid_episode_indices = flat_episodes[first_rows]

    return FramewisePchipErrors(
        indices=valid_indices,
        episode_indices=valid_episode_indices,
        pchip_error=pchip_error,
    )


def _smooth_1d_by_episode(values: Tensor, episode_indices: Tensor, window: int) -> Tensor:
    if values.ndim != 1 or episode_indices.ndim != 1:
        raise ValueError("values and episode_indices must be 1D tensors")
    if values.numel() != episode_indices.numel():
        raise ValueError("values and episode_indices must have the same length")
    if window <= 0:
        raise ValueError("smoothing_window must be positive")
    if window == 1 or values.numel() == 0:
        return values.clone()

    smoothed = torch.empty_like(values)
    left = (window - 1) // 2
    right = window // 2
    for episode_index in torch.unique(episode_indices, sorted=True):
        positions = torch.nonzero(episode_indices == episode_index, as_tuple=False).flatten()
        episode_values = values[positions]
        for row in range(episode_values.numel()):
            start = max(0, row - left)
            end = min(episode_values.numel(), row + right + 1)
            smoothed[positions[row]] = episode_values[start:end].mean()
    return smoothed


def _max_envelope_1d_by_episode(values: Tensor, episode_indices: Tensor, window: int) -> Tensor:
    """Compute a forward-looking max envelope within each episode."""
    if values.ndim != 1 or episode_indices.ndim != 1:
        raise ValueError("values and episode_indices must be 1D tensors")
    if values.numel() != episode_indices.numel():
        raise ValueError("values and episode_indices must have the same length")
    if window <= 0:
        raise ValueError("envelope_window must be positive")
    if window == 1 or values.numel() == 0:
        return values.clone()

    envelope = torch.empty_like(values)
    for episode_index in torch.unique(episode_indices, sorted=True):
        positions = torch.nonzero(episode_indices == episode_index, as_tuple=False).flatten()
        episode_values = values[positions]
        for row in range(episode_values.numel()):
            end = min(episode_values.numel(), row + window)
            envelope[positions[row]] = episode_values[row:end].max()
    return envelope


def compute_pchip_error_lambda_metrics(
    chunks: Tensor,
    episode_indices: Tensor,
    cfg: LambdaLabelConfig,
    q_low: float = 0.05,
    q_high: float = 0.95,
    envelope_window: int = 5,
    smoothing_window: int = 11,
    smoothing_alpha: float = 0.7,
) -> LambdaMetrics:
    if chunks.ndim != 3 or chunks.shape[1] != cfg.chunk_size:
        raise ValueError(f"Expected chunks shaped [N, {cfg.chunk_size}, D]")
    if not 0.0 <= smoothing_alpha <= 1.0:
        raise ValueError("smoothing_alpha must be in [0, 1]")

    episode_indices = torch.as_tensor(episode_indices, dtype=torch.long, device=chunks.device).view(-1)
    if episode_indices.numel() != chunks.shape[0]:
        raise ValueError("episode_indices must have one value per chunk")

    trend = compute_pchip_trend(chunks, cfg)
    query = list(cfg.query_indices)
    pchip_error = (chunks[:, query] - trend[:, query]).pow(2).mean(dim=(1, 2))
    lambda_raw, q_low_value, q_high_value = _normalize_by_quantiles(
        pchip_error, q_low=q_low, q_high=q_high, eps=cfg.eps
    )
    lambda_envelope = _max_envelope_1d_by_episode(lambda_raw, episode_indices, envelope_window)
    smoothed = _smooth_1d_by_episode(lambda_envelope, episode_indices, smoothing_window)
    lambda_t = torch.clamp(
        smoothing_alpha * smoothed + (1.0 - smoothing_alpha) * lambda_envelope, 0.0, 1.0
    )
    return LambdaMetrics(
        pchip_error=pchip_error,
        lambda_raw=lambda_raw,
        lambda_envelope=lambda_envelope,
        lambda_t=lambda_t,
        q_low_value=q_low_value,
        q_high_value=q_high_value,
    )


def _summarize_lambda_metrics(metrics: LambdaMetrics) -> dict[str, Tensor]:
    return {
        "pchip_error_mean": metrics.pchip_error.mean().detach(),
        "pchip_error_std": metrics.pchip_error.std(unbiased=False).detach(),
        "lambda_raw_mean": metrics.lambda_raw.mean().detach(),
        "lambda_raw_std": metrics.lambda_raw.std(unbiased=False).detach(),
        "lambda_envelope_mean": metrics.lambda_envelope.mean().detach(),
        "lambda_envelope_std": metrics.lambda_envelope.std(unbiased=False).detach(),
        "lambda_t_mean": metrics.lambda_t.mean().detach(),
        "lambda_t_std": metrics.lambda_t.std(unbiased=False).detach(),
        "q_low_value": metrics.q_low_value.detach(),
        "q_high_value": metrics.q_high_value.detach(),
    }


def _summarize_dct_metrics(
    dct_high_freq_ratio: Tensor,
    lambda_dct: Tensor,
    alpha_conf: Tensor,
    dct_q_low_value: Tensor,
    dct_q_high_value: Tensor,
) -> dict[str, Tensor]:
    return {
        "dct_high_freq_ratio_mean": dct_high_freq_ratio.mean().detach(),
        "dct_high_freq_ratio_std": dct_high_freq_ratio.std(unbiased=False).detach(),
        "lambda_dct_mean": lambda_dct.mean().detach(),
        "lambda_dct_std": lambda_dct.std(unbiased=False).detach(),
        "alpha_conf_mean": alpha_conf.mean().detach(),
        "alpha_conf_std": alpha_conf.std(unbiased=False).detach(),
        "dct_q_low_value": dct_q_low_value.detach(),
        "dct_q_high_value": dct_q_high_value.detach(),
    }


def normalize_action_chunks(
    action_chunks: Tensor,
    action_stats: dict[str, Any],
    normalization_mode: NormalizationMode,
    eps: float = 1e-8,
) -> Tensor:
    stats = {
        key: torch.as_tensor(value, dtype=action_chunks.dtype, device=action_chunks.device)
        for key, value in action_stats.items()
    }
    if normalization_mode == NormalizationMode.IDENTITY:
        return action_chunks
    if normalization_mode == NormalizationMode.MEAN_STD:
        return (action_chunks - stats["mean"]) / (stats["std"] + eps)
    if normalization_mode == NormalizationMode.MIN_MAX:
        denom = torch.where(
            stats["max"] == stats["min"],
            torch.full_like(stats["max"], eps),
            stats["max"] - stats["min"],
        )
        return 2.0 * (action_chunks - stats["min"]) / denom - 1.0
    if normalization_mode == NormalizationMode.QUANTILES:
        denom = torch.where(
            stats["q99"] == stats["q01"],
            torch.full_like(stats["q99"], eps),
            stats["q99"] - stats["q01"],
        )
        return 2.0 * (action_chunks - stats["q01"]) / denom - 1.0
    if normalization_mode == NormalizationMode.QUANTILE10:
        denom = torch.where(
            stats["q90"] == stats["q10"],
            torch.full_like(stats["q90"], eps),
            stats["q90"] - stats["q10"],
        )
        return 2.0 * (action_chunks - stats["q10"]) / denom - 1.0
    raise ValueError(f"Unsupported action normalization mode: {normalization_mode}")


def save_lambda_sidecar(
    path: str | Path,
    valid_indices: Tensor,
    lambda_t: Tensor | None = None,
    metrics: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    lambda_pchip: Tensor | None = None,
    alpha_conf: Tensor | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if lambda_pchip is None:
        if lambda_t is None:
            raise ValueError("save_lambda_sidecar requires lambda_pchip or lambda_t")
        lambda_pchip = lambda_t
    if lambda_t is None:
        lambda_t = lambda_pchip

    lambda_pchip = torch.as_tensor(lambda_pchip, dtype=torch.float32).view(-1)
    lambda_t = torch.as_tensor(lambda_t, dtype=torch.float32).view(-1)
    if alpha_conf is None:
        alpha_conf = torch.ones_like(lambda_pchip)
    alpha_conf = torch.as_tensor(alpha_conf, dtype=torch.float32).view(-1)
    valid_indices = torch.as_tensor(valid_indices, dtype=torch.long).view(-1)
    if not (valid_indices.numel() == lambda_pchip.numel() == lambda_t.numel() == alpha_conf.numel()):
        raise ValueError("valid_indices, lambda_pchip, lambda_t, and alpha_conf must have matching lengths")

    payload: dict[str, Any] = {
        "valid_indices": valid_indices.detach().cpu(),
        "lambda_pchip": lambda_pchip.detach().cpu(),
        "lambda_t": lambda_t.detach().cpu().to(dtype=torch.float32),
        "alpha_conf": alpha_conf.detach().cpu(),
        "metrics": {
            key: torch.as_tensor(value).detach().cpu() for key, value in (metrics or {}).items()
        },
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_lambda_sidecar(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "valid_indices" not in payload or ("lambda_pchip" not in payload and "lambda_t" not in payload):
        raise ValueError("Lambda sidecar must contain valid_indices and lambda_pchip or lambda_t")
    if "lambda_pchip" not in payload:
        payload["lambda_pchip"] = payload["lambda_t"]
    if "lambda_t" not in payload:
        payload["lambda_t"] = payload["lambda_pchip"]
    payload.pop("lambda_by_index", None)
    payload.pop("diagnostics", None)
    return payload


class LambdaLabelLookup:
    def __init__(self, payload: dict[str, Any], default_value: float = 0.0):
        self.default_value = float(default_value)
        indices = torch.as_tensor(payload["valid_indices"], dtype=torch.long)
        values = torch.as_tensor(payload.get("lambda_pchip", payload["lambda_t"]), dtype=torch.float32)
        confidence = torch.as_tensor(
            payload.get("alpha_conf", torch.ones_like(values, dtype=torch.float32)),
            dtype=torch.float32,
        )
        if indices.ndim != 1 or values.ndim != 1 or confidence.ndim != 1:
            raise ValueError("Lambda sidecar valid_indices, lambda_pchip, and alpha_conf must be 1D tensors")
        if not (indices.numel() == values.numel() == confidence.numel()):
            raise ValueError("Lambda sidecar valid_indices, lambda_pchip, and alpha_conf must match length")

        order = torch.argsort(indices)
        self._indices = indices[order].cpu()
        self._values = values[order].cpu()
        self._confidence = confidence[order].cpu()
        if self._indices.numel() > 1 and torch.any(self._indices[1:] == self._indices[:-1]):
            raise ValueError("Lambda sidecar valid_indices must be unique")

    def lookup(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        values, _confidence, valid = self.lookup_with_confidence(indices)
        return values, valid

    def lookup_with_confidence(self, indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        indices_tensor = torch.as_tensor(indices, dtype=torch.long)
        flat_indices = indices_tensor.view(-1).cpu()
        values = torch.full((flat_indices.numel(),), self.default_value, dtype=torch.float32)
        confidence = torch.zeros((flat_indices.numel(),), dtype=torch.float32)
        valid = torch.zeros((flat_indices.numel(),), dtype=torch.bool)
        if self._indices.numel() == 0:
            return (
                values.view(indices_tensor.shape),
                confidence.view(indices_tensor.shape),
                valid.view(indices_tensor.shape),
            )

        positions = torch.searchsorted(self._indices, flat_indices)
        in_bounds = positions < self._indices.numel()
        safe_positions = positions.clamp(max=self._indices.numel() - 1)
        valid = in_bounds & (self._indices[safe_positions] == flat_indices)
        values[valid] = self._values[safe_positions[valid]]
        confidence[valid] = self._confidence[safe_positions[valid]]
        return (
            values.view(indices_tensor.shape),
            confidence.view(indices_tensor.shape),
            valid.view(indices_tensor.shape),
        )


def _episode_value(episode: dict[str, Any], key: str) -> int:
    value = episode[key]
    if isinstance(value, list):
        value = value[0]
    if torch.is_tensor(value):
        value = value.item()
    return int(value)


def _stack_action_column(values: Any) -> Tensor:
    tensors = [torch.as_tensor(value, dtype=torch.float32) for value in values]
    if not tensors:
        return torch.empty((0,), dtype=torch.float32)
    return torch.stack(tensors, dim=0)


def _scalar_column_to_long(values: Any) -> Tensor:
    return torch.tensor(
        [int(value.item() if torch.is_tensor(value) else value) for value in values],
        dtype=torch.long,
    )


def _preload_action_columns(dataset: Any) -> PreloadedActionColumns:
    action_dataset = dataset.select_columns([ACTION, "index", "episode_index"])
    actions = _stack_action_column(action_dataset[ACTION])
    indices = _scalar_column_to_long(action_dataset["index"])
    episode_indices = _scalar_column_to_long(action_dataset["episode_index"])
    if not (actions.shape[0] == indices.numel() == episode_indices.numel()):
        raise ValueError("Preloaded action columns have inconsistent lengths")
    return PreloadedActionColumns(
        indices=indices,
        episode_indices=episode_indices,
        actions=actions,
        row_by_index={int(index): row for row, index in enumerate(indices.tolist())},
    )


def extract_action_chunks_from_dataset(dataset: Any, cfg: LambdaLabelConfig) -> ActionChunkBatch:
    action_columns = _preload_action_columns(dataset)
    indices: list[int] = []
    chunk_episode_indices: list[int] = []
    chunks: list[Tensor] = []
    for ep_idx in range(dataset.num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        from_idx = _episode_value(ep, "dataset_from_index")
        to_idx = _episode_value(ep, "dataset_to_index")
        if to_idx - from_idx < cfg.chunk_size:
            continue
        episode_actions: list[Tensor] = []
        episode_start_indices: list[int] = []
        episode_indices: list[int] = []
        for abs_idx in range(from_idx, to_idx):
            if dataset.reader._absolute_to_relative_idx is not None:
                rel_idx = dataset.reader._absolute_to_relative_idx.get(abs_idx)
                if rel_idx is None:
                    break
            else:
                rel_idx = action_columns.row_by_index.get(abs_idx)
                if rel_idx is None:
                    break
            episode_start_indices.append(int(action_columns.indices[rel_idx].item()))
            episode_indices.append(int(action_columns.episode_indices[rel_idx].item()))
            episode_actions.append(action_columns.actions[rel_idx])
        if len(episode_actions) < cfg.chunk_size:
            continue
        actions = torch.stack(episode_actions, dim=0)
        for start in range(0, actions.shape[0] - cfg.chunk_size + 1):
            indices.append(episode_start_indices[start])
            chunk_episode_indices.append(episode_indices[start])
            chunks.append(actions[start : start + cfg.chunk_size])
    if not chunks:
        raise ValueError("No valid action chunks were found in the dataset")
    return ActionChunkBatch(
        indices=torch.tensor(indices, dtype=torch.long),
        episode_indices=torch.tensor(chunk_episode_indices, dtype=torch.long),
        chunks=torch.stack(chunks, dim=0),
    )


@torch.no_grad()
def generate_lambda_labels_from_chunks(
    chunks: Tensor,
    indices: Tensor,
    episode_indices: Tensor,
    cfg: LambdaLabelConfig,
    q_low: float = 0.05,
    q_high: float = 0.95,
    dct_q_low: float | None = None,
    dct_q_high: float | None = None,
    dct_high_freq_start: int | None = None,
    alpha_conf_min: float = 0.15,
    alpha_conf_max: float = 1.0,
    envelope_window: int = 5,
    smoothing_window: int = 11,
    smoothing_alpha: float = 0.7,
) -> dict[str, Any]:
    indices = torch.as_tensor(indices, dtype=torch.long).view(-1)
    episode_indices = torch.as_tensor(episode_indices, dtype=torch.long).view(-1)
    if chunks.shape[0] != indices.numel() or chunks.shape[0] != episode_indices.numel():
        raise ValueError("chunks, indices, and episode_indices must have matching first dimensions")

    metrics = compute_pchip_error_lambda_metrics(
        chunks=chunks,
        episode_indices=episode_indices,
        cfg=cfg,
        q_low=q_low,
        q_high=q_high,
        envelope_window=envelope_window,
        smoothing_window=smoothing_window,
        smoothing_alpha=smoothing_alpha,
    )
    dct_high_freq_ratio = compute_dct_high_frequency_ratio(
        chunks,
        high_freq_start=dct_high_freq_start,
        eps=cfg.eps,
    )
    lambda_dct_raw, dct_q_low_value, dct_q_high_value = _normalize_by_quantiles(
        dct_high_freq_ratio,
        q_low=q_low if dct_q_low is None else dct_q_low,
        q_high=q_high if dct_q_high is None else dct_q_high,
        eps=cfg.eps,
    )
    lambda_dct = _max_envelope_1d_by_episode(lambda_dct_raw, episode_indices, envelope_window)
    alpha_conf = compute_lambda_confidence(
        lambda_pchip=metrics.lambda_t,
        lambda_dct=lambda_dct,
        alpha_conf_min=alpha_conf_min,
        alpha_conf_max=alpha_conf_max,
    )
    metric_summary = _summarize_lambda_metrics(metrics)
    metric_summary.update(
        _summarize_dct_metrics(
            dct_high_freq_ratio=dct_high_freq_ratio,
            lambda_dct=lambda_dct,
            alpha_conf=alpha_conf,
            dct_q_low_value=dct_q_low_value,
            dct_q_high_value=dct_q_high_value,
        )
    )
    return {
        "valid_indices": indices.detach().cpu(),
        "episode_indices": episode_indices.detach().cpu(),
        "lambda_pchip": metrics.lambda_t.detach().cpu().to(dtype=torch.float32),
        "lambda_t": metrics.lambda_t.detach().cpu().to(dtype=torch.float32),
        "alpha_conf": alpha_conf.detach().cpu().to(dtype=torch.float32),
        "metrics": metric_summary,
        "pchip_error": metrics.pchip_error.detach().cpu().to(dtype=torch.float32),
        "lambda_raw": metrics.lambda_raw.detach().cpu().to(dtype=torch.float32),
        "lambda_envelope": metrics.lambda_envelope.detach().cpu().to(dtype=torch.float32),
        "dct_high_freq_ratio": dct_high_freq_ratio.detach().cpu().to(dtype=torch.float32),
        "lambda_dct": lambda_dct.detach().cpu().to(dtype=torch.float32),
    }


def _format_csv_float(value: Tensor | float) -> str:
    return str(round(float(value), 6))


def _parse_position_weights(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    weights = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not weights:
        raise ValueError("lambda-position-weights must contain at least one numeric value")
    return weights


def save_lambda_diagnostics_csv(
    path: str | Path,
    indices: Tensor,
    episode_indices: Tensor,
    pchip_error: Tensor,
    lambda_raw: Tensor,
    lambda_pchip: Tensor,
    dct_high_freq_ratio: Tensor,
    lambda_dct: Tensor,
    alpha_conf: Tensor,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = zip(
        torch.as_tensor(indices, dtype=torch.long).view(-1).tolist(),
        torch.as_tensor(episode_indices, dtype=torch.long).view(-1).tolist(),
        torch.as_tensor(pchip_error, dtype=torch.float32).view(-1),
        torch.as_tensor(lambda_raw, dtype=torch.float32).view(-1),
        torch.as_tensor(lambda_pchip, dtype=torch.float32).view(-1),
        torch.as_tensor(dct_high_freq_ratio, dtype=torch.float32).view(-1),
        torch.as_tensor(lambda_dct, dtype=torch.float32).view(-1),
        torch.as_tensor(alpha_conf, dtype=torch.float32).view(-1),
        strict=True,
    )
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "index",
                "episode_index",
                "pchip_error",
                "lambda_raw",
                "lambda_pchip",
                "dct_high_freq_ratio",
                "lambda_dct",
                "alpha_conf",
            ],
        )
        writer.writeheader()
        for index, episode_index, error, raw_value, value, dct_ratio, dct_value, confidence in rows:
            writer.writerow(
                {
                    "index": int(index),
                    "episode_index": int(episode_index),
                    "pchip_error": _format_csv_float(error),
                    "lambda_raw": _format_csv_float(raw_value),
                    "lambda_pchip": _format_csv_float(value),
                    "dct_high_freq_ratio": _format_csv_float(dct_ratio),
                    "lambda_dct": _format_csv_float(dct_value),
                    "alpha_conf": _format_csv_float(confidence),
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SmolVLA PCHIP lambda labels with DCT confidence.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--normalization-mode", default=NormalizationMode.MEAN_STD.value)
    parser.add_argument("--lambda-error-q-low", type=float, default=0.05)
    parser.add_argument("--lambda-error-q-high", type=float, default=0.95)
    parser.add_argument("--lambda-dct-q-low", type=float, default=None)
    parser.add_argument("--lambda-dct-q-high", type=float, default=None)
    parser.add_argument("--lambda-dct-high-freq-start", type=int, default=None)
    parser.add_argument("--lambda-alpha-conf-min", type=float, default=0.15)
    parser.add_argument("--lambda-alpha-conf-max", type=float, default=1.0)
    parser.add_argument("--lambda-envelope-window", type=int, default=5)
    parser.add_argument("--lambda-smoothing-window", type=int, default=11)
    parser.add_argument("--lambda-smoothing-alpha", type=float, default=0.7)
    parser.add_argument(
        "--lambda-position-weights",
        default=None,
        help="Comma-separated per-position aggregation weights. Defaults to 1 for every chunk position.",
    )
    parser.add_argument("--diagnostics-path", default=None)
    return parser


def run_offline_lambda_label_generation(args: argparse.Namespace) -> None:
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, revision=args.revision)
    cfg = LambdaLabelConfig(position_weights=_parse_position_weights(args.lambda_position_weights))
    chunk_batch = extract_action_chunks_from_dataset(dataset, cfg)
    action_norm_mode = NormalizationMode(args.normalization_mode)
    normalized_chunks = normalize_action_chunks(
        chunk_batch.chunks, dataset.meta.stats[ACTION], action_norm_mode
    )
    labels = generate_lambda_labels_from_chunks(
        chunks=normalized_chunks,
        indices=chunk_batch.indices,
        episode_indices=chunk_batch.episode_indices,
        cfg=cfg,
        q_low=args.lambda_error_q_low,
        q_high=args.lambda_error_q_high,
        dct_q_low=args.lambda_dct_q_low,
        dct_q_high=args.lambda_dct_q_high,
        dct_high_freq_start=args.lambda_dct_high_freq_start,
        alpha_conf_min=args.lambda_alpha_conf_min,
        alpha_conf_max=args.lambda_alpha_conf_max,
        envelope_window=args.lambda_envelope_window,
        smoothing_window=args.lambda_smoothing_window,
        smoothing_alpha=args.lambda_smoothing_alpha,
    )
    metadata = {
        "repo_id": args.repo_id,
        "root": args.root,
        "revision": args.revision,
        "created_at": datetime.now(UTC).isoformat(),
        "method": "pchip_dct_consistency",
        "chunk_size": cfg.chunk_size,
        "anchor_indices": list(cfg.anchor_indices),
        "query_indices": list(cfg.query_indices),
        "position_weights": list(_position_weights_tensor(cfg, torch.device("cpu"), torch.float32).tolist()),
        "normalization_mode": action_norm_mode.value,
        "lambda_error_q_low": args.lambda_error_q_low,
        "lambda_error_q_high": args.lambda_error_q_high,
        "lambda_dct_q_low": args.lambda_dct_q_low,
        "lambda_dct_q_high": args.lambda_dct_q_high,
        "lambda_dct_high_freq_start": args.lambda_dct_high_freq_start,
        "lambda_alpha_conf_min": args.lambda_alpha_conf_min,
        "lambda_alpha_conf_max": args.lambda_alpha_conf_max,
        "lambda_envelope_window": args.lambda_envelope_window,
        "lambda_smoothing_window": args.lambda_smoothing_window,
        "lambda_smoothing_alpha": args.lambda_smoothing_alpha,
    }
    save_lambda_sidecar(
        path=args.output_path,
        valid_indices=labels["valid_indices"],
        lambda_pchip=labels["lambda_pchip"],
        lambda_t=labels["lambda_t"],
        alpha_conf=labels["alpha_conf"],
        metrics=labels["metrics"],
        metadata=metadata,
    )
    if args.diagnostics_path is not None:
        save_lambda_diagnostics_csv(
            path=args.diagnostics_path,
            indices=labels["valid_indices"],
            episode_indices=labels["episode_indices"],
            pchip_error=labels["pchip_error"],
            lambda_raw=labels["lambda_raw"],
            lambda_pchip=labels["lambda_pchip"],
            dct_high_freq_ratio=labels["dct_high_freq_ratio"],
            lambda_dct=labels["lambda_dct"],
            alpha_conf=labels["alpha_conf"],
        )


def main() -> None:
    parser = build_arg_parser()
    run_offline_lambda_label_generation(parser.parse_args())


if __name__ == "__main__":
    main()
