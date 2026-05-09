#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from lerobot.configs.types import NormalizationMode
from lerobot.utils.constants import ACTION


@dataclass(frozen=True)
class LambdaLabelConfig:
    chunk_size: int = 10
    anchor_indices: tuple[int, ...] = (0, 3, 6, 9)
    query_indices: tuple[int, ...] = (1, 2, 4, 5, 7, 8)
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


class LambdaMetrics(NamedTuple):
    a_hat: Tensor
    e_trend: Tensor
    e_full: Tensor
    e_improve: Tensor
    lambda_t: Tensor


class ActionChunkBatch(NamedTuple):
    indices: Tensor
    chunks: Tensor


@dataclass
class ResidualTrainingConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 256
    device: str = "cpu"


class MaskedResidualPredictor(nn.Module):
    def __init__(self, chunk_size: int, action_dim: int, num_query_positions: int, hidden_dim: int = 256):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.num_query_positions = num_query_positions
        self.net = nn.Sequential(
            nn.Linear(chunk_size * action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_query_positions * action_dim),
        )

    def forward(self, trend: Tensor) -> Tensor:
        if trend.ndim != 3:
            raise ValueError("trend must have shape [N, chunk_size, action_dim]")
        out = self.net(trend.flatten(start_dim=1))
        return out.view(trend.shape[0], self.num_query_positions, self.action_dim)


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


def scatter_query_residuals(query_residuals: Tensor, action_dim: int, cfg: LambdaLabelConfig) -> Tensor:
    if query_residuals.ndim != 3:
        raise ValueError("query_residuals must have shape [N, len(query_indices), D]")
    if query_residuals.shape[1] != len(cfg.query_indices):
        raise ValueError("query_residuals second dimension must match query_indices")
    if query_residuals.shape[2] != action_dim:
        raise ValueError("query_residuals action dimension must match action_dim")

    residuals = query_residuals.new_zeros(query_residuals.shape[0], cfg.chunk_size, action_dim)
    residuals[:, list(cfg.query_indices), :] = query_residuals
    return residuals


def compute_lambda_metrics(
    a_t: Tensor, a_trend: Tensor, a_residual: Tensor, cfg: LambdaLabelConfig
) -> LambdaMetrics:
    if a_t.shape != a_trend.shape or a_t.shape != a_residual.shape:
        raise ValueError("a_t, a_trend, and a_residual must have identical shapes")
    if a_t.ndim != 3 or a_t.shape[1] != cfg.chunk_size:
        raise ValueError(f"Expected tensors shaped [N, {cfg.chunk_size}, D]")

    a_hat = a_trend + a_residual
    query = list(cfg.query_indices)
    e_trend = (a_t[:, query, :] - a_trend[:, query, :]).pow(2).mean(dim=(1, 2))
    e_full = (a_t[:, query, :] - a_hat[:, query, :]).pow(2).mean(dim=(1, 2))
    e_improve = e_trend - e_full
    lambda_t = torch.clamp(e_improve / (e_trend + cfg.eps), min=0.0, max=1.0)
    return LambdaMetrics(a_hat=a_hat, e_trend=e_trend, e_full=e_full, e_improve=e_improve, lambda_t=lambda_t)


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
    lambda_t: Tensor,
    metrics: dict[str, Tensor],
    metadata: dict[str, Any],
    diagnostics: dict[str, Tensor] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "valid_indices": valid_indices.detach().cpu().to(dtype=torch.long),
        "lambda_t": lambda_t.detach().cpu().to(dtype=torch.float32),
        "metrics": {key: value.detach().cpu() for key, value in metrics.items()},
        "metadata": metadata,
    }
    payload["lambda_by_index"] = {
        int(idx): float(value)
        for idx, value in zip(payload["valid_indices"].tolist(), payload["lambda_t"].tolist(), strict=True)
    }
    if diagnostics is not None:
        payload["diagnostics"] = {key: value.detach().cpu() for key, value in diagnostics.items()}
    torch.save(payload, path)


def load_lambda_sidecar(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "valid_indices" not in payload or "lambda_t" not in payload:
        raise ValueError("Lambda sidecar must contain valid_indices and lambda_t")
    return payload


class LambdaLabelLookup:
    def __init__(self, payload: dict[str, Any], default_value: float = 0.0):
        self.default_value = float(default_value)
        indices = torch.as_tensor(payload["valid_indices"], dtype=torch.long)
        values = torch.as_tensor(payload["lambda_t"], dtype=torch.float32)
        self._values_by_index = {
            int(idx): float(value) for idx, value in zip(indices.tolist(), values.tolist(), strict=True)
        }

    def lookup(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        flat_indices = torch.as_tensor(indices, dtype=torch.long).view(-1).cpu()
        values = torch.full((flat_indices.numel(),), self.default_value, dtype=torch.float32)
        valid = torch.zeros((flat_indices.numel(),), dtype=torch.bool)
        for row, idx in enumerate(flat_indices.tolist()):
            if idx in self._values_by_index:
                values[row] = self._values_by_index[idx]
                valid[row] = True
        return values.view(indices.shape), valid.view(indices.shape)


def _episode_value(episode: dict[str, Any], key: str) -> int:
    value = episode[key]
    if isinstance(value, list):
        value = value[0]
    if torch.is_tensor(value):
        value = value.item()
    return int(value)


def extract_action_chunks_from_dataset(dataset: Any, cfg: LambdaLabelConfig) -> ActionChunkBatch:
    indices: list[int] = []
    chunks: list[Tensor] = []
    for ep_idx in range(dataset.num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        from_idx = _episode_value(ep, "dataset_from_index")
        to_idx = _episode_value(ep, "dataset_to_index")
        if to_idx - from_idx < cfg.chunk_size:
            continue
        episode_actions: list[Tensor] = []
        episode_indices: list[int] = []
        for abs_idx in range(from_idx, to_idx):
            if dataset.reader._absolute_to_relative_idx is not None:
                rel_idx = dataset.reader._absolute_to_relative_idx.get(abs_idx)
                if rel_idx is None:
                    break
            else:
                rel_idx = abs_idx
            frame = dataset.get_raw_item(rel_idx)
            episode_indices.append(int(torch.as_tensor(frame["index"]).item()))
            episode_actions.append(torch.as_tensor(frame[ACTION], dtype=torch.float32))
        if len(episode_actions) < cfg.chunk_size:
            continue
        actions = torch.stack(episode_actions, dim=0)
        for start in range(0, actions.shape[0] - cfg.chunk_size + 1):
            indices.append(episode_indices[start])
            chunks.append(actions[start : start + cfg.chunk_size])
    if not chunks:
        raise ValueError("No valid action chunks were found in the dataset")
    return ActionChunkBatch(
        indices=torch.tensor(indices, dtype=torch.long), chunks=torch.stack(chunks, dim=0)
    )


def train_residual_predictor(
    normalized_chunks: Tensor,
    cfg: LambdaLabelConfig,
    train_cfg: ResidualTrainingConfig,
) -> MaskedResidualPredictor:
    trend = compute_pchip_trend(normalized_chunks, cfg)
    target = normalized_chunks[:, list(cfg.query_indices), :] - trend[:, list(cfg.query_indices), :]
    predictor = MaskedResidualPredictor(
        chunk_size=cfg.chunk_size,
        action_dim=normalized_chunks.shape[-1],
        num_query_positions=len(cfg.query_indices),
        hidden_dim=train_cfg.hidden_dim,
    ).to(train_cfg.device)
    dataset = TensorDataset(trend.to(torch.float32), target.to(torch.float32))
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=train_cfg.lr)
    predictor.train()
    for _epoch in range(train_cfg.epochs):
        for batch_trend, batch_target in loader:
            batch_trend = batch_trend.to(train_cfg.device)
            batch_target = batch_target.to(train_cfg.device)
            pred = predictor(batch_trend)
            loss = torch.nn.functional.mse_loss(pred, batch_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    predictor.eval()
    return predictor


@torch.no_grad()
def generate_lambda_labels_from_chunks(
    chunks: Tensor,
    indices: Tensor,
    predictor: nn.Module,
    cfg: LambdaLabelConfig,
    batch_size: int = 256,
    save_diagnostics: bool = False,
    max_diagnostic_chunks: int | None = None,
) -> dict[str, Any]:
    predictor.eval()
    trend = compute_pchip_trend(chunks, cfg)
    residual_batches: list[Tensor] = []
    try:
        device = next(predictor.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    for start in range(0, chunks.shape[0], batch_size):
        batch_trend = trend[start : start + batch_size].to(device)
        query_residuals = predictor(batch_trend).cpu()
        residual_batches.append(scatter_query_residuals(query_residuals, chunks.shape[-1], cfg))
    residuals = torch.cat(residual_batches, dim=0).to(dtype=chunks.dtype)
    metrics = compute_lambda_metrics(chunks, trend, residuals, cfg)
    payload: dict[str, Any] = {
        "valid_indices": indices.detach().cpu().to(dtype=torch.long),
        "lambda_t": metrics.lambda_t.detach().cpu().to(dtype=torch.float32),
        "metrics": {
            "e_trend": metrics.e_trend.detach().cpu(),
            "e_full": metrics.e_full.detach().cpu(),
            "e_improve": metrics.e_improve.detach().cpu(),
        },
    }
    if save_diagnostics:
        limit = (
            chunks.shape[0] if max_diagnostic_chunks is None else min(max_diagnostic_chunks, chunks.shape[0])
        )
        payload["diagnostics"] = {
            "A_t": chunks[:limit].detach().cpu(),
            "A_trend": trend[:limit].detach().cpu(),
            "A_residual": residuals[:limit].detach().cpu(),
            "A_hat": metrics.a_hat[:limit].detach().cpu(),
        }
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SmolVLA residual predictor and generate lambda labels."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--normalization-mode", default=NormalizationMode.MEAN_STD.value)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument("--max-diagnostic-chunks", type=int, default=128)
    return parser


def run_offline_lambda_label_generation(args: argparse.Namespace) -> None:
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, revision=args.revision)
    cfg = LambdaLabelConfig()
    chunk_batch = extract_action_chunks_from_dataset(dataset, cfg)
    action_norm_mode = NormalizationMode(args.normalization_mode)
    normalized_chunks = normalize_action_chunks(
        chunk_batch.chunks, dataset.meta.stats[ACTION], action_norm_mode
    )
    train_cfg = ResidualTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    predictor = train_residual_predictor(normalized_chunks, cfg, train_cfg)
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": predictor.state_dict(), "config": train_cfg.__dict__}, checkpoint_path
        )
    labels = generate_lambda_labels_from_chunks(
        chunks=normalized_chunks,
        indices=chunk_batch.indices,
        predictor=predictor,
        cfg=cfg,
        batch_size=args.batch_size,
        save_diagnostics=args.save_diagnostics,
        max_diagnostic_chunks=args.max_diagnostic_chunks,
    )
    metadata = {
        "repo_id": args.repo_id,
        "root": args.root,
        "revision": args.revision,
        "created_at": datetime.now(UTC).isoformat(),
        "chunk_size": cfg.chunk_size,
        "anchor_indices": list(cfg.anchor_indices),
        "query_indices": list(cfg.query_indices),
        "normalization_mode": action_norm_mode.value,
        "checkpoint_path": args.checkpoint_path,
    }
    save_lambda_sidecar(
        path=args.output_path,
        valid_indices=labels["valid_indices"],
        lambda_t=labels["lambda_t"],
        metrics=labels["metrics"],
        metadata=metadata,
        diagnostics=labels.get("diagnostics"),
    )


def main() -> None:
    parser = build_arg_parser()
    run_offline_lambda_label_generation(parser.parse_args())


if __name__ == "__main__":
    main()
