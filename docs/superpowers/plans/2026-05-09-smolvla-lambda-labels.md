# SmolVLA Lambda Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable SmolVLA offline lambda-label pipeline and connect those labels to SmolVLA training and dynamic inference.

**Architecture:** Offline code in `src/lerobot/policies/smolvla/lambda_labels.py` creates normalized 10-step action chunks, computes PCHIP trends, trains a masked residual predictor, and writes a sidecar label file keyed by dataset `index`. SmolVLA config and preprocessing load that sidecar into `lambda_t`/`lambda_is_valid`; the flow model predicts `lambda_hat`, conditions the action expert with a lambda token, adds an auxiliary loss, and optionally uses predicted lambda to choose a shorter action execution horizon at inference.

**Tech Stack:** Python 3.12, PyTorch, SciPy PCHIP, LeRobotDataset, draccus/argparse-style CLI, pytest, uv.

---

## File Structure

- Create `src/lerobot/policies/smolvla/lambda_labels.py`
  - Owns offline label config validation, action normalization, PCHIP trend construction, residual predictor, two-stage train/generate workflow, sidecar save/load, and a module entrypoint runnable with `python -m lerobot.policies.smolvla.lambda_labels`.
- Modify `src/lerobot/policies/smolvla/configuration_smolvla.py`
  - Adds lambda label, lambda loss, lambda token, alpha warmup, and dynamic execution settings.
- Modify `src/lerobot/policies/smolvla/processor_smolvla.py`
  - Adds SmolVLA-specific sidecar lookup processor and inserts it when `config.lambda_labels_path` is set.
- Modify `src/lerobot/policies/smolvla/modeling_smolvla.py`
  - Adds lambda prediction head, lambda token MLP, lambda-conditioned suffix path, lambda auxiliary loss plumbing, train-step buffer, and dynamic execution helpers.
- Modify `pyproject.toml`
  - Adds a console script for the offline tool.
- Create `tests/policies/smolvla/test_lambda_labels.py`
  - Covers PCHIP, residual masking, lambda metrics, sidecar lookup, and lightweight generation.
- Modify `tests/processor/test_smolvla_processor.py`
  - Covers sidecar label injection into batches.
- Create `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`
  - Covers lambda suffix shape, alpha/condition computation, and dynamic execution math without loading real VLM weights.

---

### Task 1: Lambda Math Utilities

**Files:**
- Create: `src/lerobot/policies/smolvla/lambda_labels.py`
- Create: `tests/policies/smolvla/test_lambda_labels.py`

- [ ] **Step 1: Write failing tests for PCHIP, residual scattering, and lambda metrics**

Add this file:

```python
import pytest
import torch

from lerobot.policies.smolvla.lambda_labels import (
    LambdaLabelConfig,
    compute_lambda_metrics,
    compute_pchip_trend,
    scatter_query_residuals,
)


def test_compute_pchip_trend_preserves_anchor_values():
    pytest.importorskip("scipy")
    cfg = LambdaLabelConfig(chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8))
    chunks = torch.zeros(2, 10, 3)
    chunks[:, 0] = torch.tensor([0.0, 1.0, 2.0])
    chunks[:, 3] = torch.tensor([3.0, 4.0, 5.0])
    chunks[:, 6] = torch.tensor([6.0, 7.0, 8.0])
    chunks[:, 9] = torch.tensor([9.0, 10.0, 11.0])

    trend = compute_pchip_trend(chunks, cfg)

    assert trend.shape == chunks.shape
    assert torch.allclose(trend[:, cfg.anchor_indices], chunks[:, cfg.anchor_indices], atol=1e-6)


def test_scatter_query_residuals_keeps_anchor_residual_zero():
    cfg = LambdaLabelConfig(chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8))
    query_residuals = torch.ones(4, len(cfg.query_indices), 2)

    full_residuals = scatter_query_residuals(query_residuals, action_dim=2, cfg=cfg)

    assert full_residuals.shape == (4, 10, 2)
    assert torch.equal(full_residuals[:, cfg.anchor_indices], torch.zeros(4, len(cfg.anchor_indices), 2))
    assert torch.equal(full_residuals[:, cfg.query_indices], torch.ones(4, len(cfg.query_indices), 2))


def test_compute_lambda_metrics_uses_query_positions_and_clamps():
    cfg = LambdaLabelConfig(chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8))
    chunks = torch.zeros(1, 10, 1)
    trend = torch.zeros_like(chunks)
    residuals = torch.zeros_like(chunks)

    chunks[:, cfg.anchor_indices] = 100.0
    chunks[:, cfg.query_indices] = 1.0
    residuals[:, cfg.query_indices] = 0.5

    metrics = compute_lambda_metrics(chunks, trend, residuals, cfg)

    assert torch.allclose(metrics.a_hat[:, cfg.query_indices], torch.full((1, 6, 1), 0.5))
    assert torch.allclose(metrics.e_trend, torch.tensor([1.0]))
    assert torch.allclose(metrics.e_full, torch.tensor([0.25]))
    assert torch.allclose(metrics.e_improve, torch.tensor([0.75]))
    assert torch.allclose(metrics.lambda_t, torch.tensor([0.75]))


def test_compute_lambda_metrics_clamps_negative_improvement_to_zero():
    cfg = LambdaLabelConfig(chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8))
    chunks = torch.zeros(1, 10, 1)
    trend = torch.zeros_like(chunks)
    residuals = torch.zeros_like(chunks)
    chunks[:, cfg.query_indices] = 1.0
    residuals[:, cfg.query_indices] = -2.0

    metrics = compute_lambda_metrics(chunks, trend, residuals, cfg)

    assert metrics.e_full.item() > metrics.e_trend.item()
    assert metrics.lambda_t.item() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py -q
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'lerobot.policies.smolvla.lambda_labels'`.

- [ ] **Step 3: Add minimal lambda math implementation**

Create `src/lerobot/policies/smolvla/lambda_labels.py` with these definitions:

```python
#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn


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


def compute_lambda_metrics(a_t: Tensor, a_trend: Tensor, a_residual: Tensor, cfg: LambdaLabelConfig) -> LambdaMetrics:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py -q
```

Expected: PASS, with PCHIP test skipped if SciPy is not installed.

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/policies/smolvla/lambda_labels.py tests/policies/smolvla/test_lambda_labels.py
git commit -m "feat: add smolvla lambda math utilities"
```

---

### Task 2: Sidecar Lookup, Normalization, and Chunk Extraction

**Files:**
- Modify: `src/lerobot/policies/smolvla/lambda_labels.py`
- Modify: `tests/policies/smolvla/test_lambda_labels.py`

- [ ] **Step 1: Add failing tests for normalization and sidecar lookup**

Append to `tests/policies/smolvla/test_lambda_labels.py`:

```python
from lerobot.configs.types import NormalizationMode
from lerobot.policies.smolvla.lambda_labels import (
    LambdaLabelLookup,
    load_lambda_sidecar,
    normalize_action_chunks,
    save_lambda_sidecar,
)


def test_normalize_action_chunks_mean_std():
    chunks = torch.tensor([[[2.0, 6.0], [4.0, 10.0]]])
    stats = {"mean": torch.tensor([1.0, 2.0]), "std": torch.tensor([1.0, 4.0])}

    normalized = normalize_action_chunks(chunks, stats, NormalizationMode.MEAN_STD)

    assert torch.allclose(normalized, torch.tensor([[[1.0, 1.0], [3.0, 2.0]]]))


def test_normalize_action_chunks_min_max():
    chunks = torch.tensor([[[0.0], [5.0], [10.0]]])
    stats = {"min": torch.tensor([0.0]), "max": torch.tensor([10.0])}

    normalized = normalize_action_chunks(chunks, stats, NormalizationMode.MIN_MAX)

    assert torch.allclose(normalized, torch.tensor([[[-1.0], [0.0], [1.0]]]))


def test_lambda_sidecar_lookup_matches_exact_start_indices_only(tmp_path):
    path = tmp_path / "labels.pt"
    save_lambda_sidecar(
        path=path,
        valid_indices=torch.tensor([10, 12]),
        lambda_t=torch.tensor([0.25, 0.75]),
        metrics={"e_trend": torch.tensor([1.0, 2.0])},
        metadata={"repo_id": "unit/test"},
    )

    loaded = load_lambda_sidecar(path)
    lookup = LambdaLabelLookup(loaded, default_value=0.1)
    values, valid = lookup.lookup(torch.tensor([9, 10, 11, 12]))

    assert torch.allclose(values, torch.tensor([0.1, 0.25, 0.1, 0.75]))
    assert torch.equal(valid, torch.tensor([False, True, False, True]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py -q
```

Expected: FAIL with import errors for `normalize_action_chunks`, `save_lambda_sidecar`, `load_lambda_sidecar`, or `LambdaLabelLookup`.

- [ ] **Step 3: Add normalization and sidecar code**

Append these definitions to `src/lerobot/policies/smolvla/lambda_labels.py`:

```python
from pathlib import Path
from typing import Any

from lerobot.configs.types import NormalizationMode


def normalize_action_chunks(
    action_chunks: Tensor,
    action_stats: dict[str, Any],
    normalization_mode: NormalizationMode,
    eps: float = 1e-8,
) -> Tensor:
    stats = {key: torch.as_tensor(value, dtype=action_chunks.dtype, device=action_chunks.device) for key, value in action_stats.items()}
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
    payload = torch.load(Path(path), map_location="cpu")
    if "valid_indices" not in payload or "lambda_t" not in payload:
        raise ValueError("Lambda sidecar must contain valid_indices and lambda_t")
    return payload


class LambdaLabelLookup:
    def __init__(self, payload: dict[str, Any], default_value: float = 0.0):
        self.default_value = float(default_value)
        indices = torch.as_tensor(payload["valid_indices"], dtype=torch.long)
        values = torch.as_tensor(payload["lambda_t"], dtype=torch.float32)
        self._values_by_index = {int(idx): float(value) for idx, value in zip(indices.tolist(), values.tolist(), strict=True)}

    def lookup(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        flat_indices = torch.as_tensor(indices, dtype=torch.long).view(-1).cpu()
        values = torch.full((flat_indices.numel(),), self.default_value, dtype=torch.float32)
        valid = torch.zeros((flat_indices.numel(),), dtype=torch.bool)
        for row, idx in enumerate(flat_indices.tolist()):
            if idx in self._values_by_index:
                values[row] = self._values_by_index[idx]
                valid[row] = True
        return values.view(indices.shape), valid.view(indices.shape)
```

- [ ] **Step 4: Add dataset chunk extraction code**

Append this extraction helper to `src/lerobot/policies/smolvla/lambda_labels.py`:

```python
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION


class ActionChunkBatch(NamedTuple):
    indices: Tensor
    chunks: Tensor


def extract_action_chunks_from_dataset(dataset: LeRobotDataset, cfg: LambdaLabelConfig) -> ActionChunkBatch:
    indices: list[int] = []
    chunks: list[Tensor] = []
    for ep_idx in range(dataset.num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])
        if to_idx - from_idx < cfg.chunk_size:
            continue
        episode_actions: list[Tensor] = []
        episode_indices: list[int] = []
        for abs_idx in range(from_idx, to_idx):
            rel_idx = dataset.reader._absolute_to_relative_idx.get(abs_idx, abs_idx) if dataset.reader._absolute_to_relative_idx is not None else abs_idx
            frame = dataset.get_raw_item(rel_idx)
            episode_indices.append(int(torch.as_tensor(frame["index"]).item()))
            episode_actions.append(torch.as_tensor(frame[ACTION], dtype=torch.float32))
        actions = torch.stack(episode_actions, dim=0)
        for start in range(0, actions.shape[0] - cfg.chunk_size + 1):
            indices.append(episode_indices[start])
            chunks.append(actions[start : start + cfg.chunk_size])
    if not chunks:
        raise ValueError("No valid action chunks were found in the dataset")
    return ActionChunkBatch(indices=torch.tensor(indices, dtype=torch.long), chunks=torch.stack(chunks, dim=0))
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py -q
```

Expected: PASS, with PCHIP test skipped if SciPy is not installed.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/policies/smolvla/lambda_labels.py tests/policies/smolvla/test_lambda_labels.py
git commit -m "feat: add smolvla lambda sidecar utilities"
```

---

### Task 3: Residual Predictor and Offline Train/Generate Tool

**Files:**
- Modify: `src/lerobot/policies/smolvla/lambda_labels.py`
- Modify: `tests/policies/smolvla/test_lambda_labels.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add failing tests for residual predictor and generation**

Append to `tests/policies/smolvla/test_lambda_labels.py`:

```python
from lerobot.policies.smolvla.lambda_labels import (
    MaskedResidualPredictor,
    generate_lambda_labels_from_chunks,
)


def test_masked_residual_predictor_outputs_query_residual_shape():
    cfg = LambdaLabelConfig(chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8))
    predictor = MaskedResidualPredictor(chunk_size=10, action_dim=4, num_query_positions=6, hidden_dim=16)
    trend = torch.randn(5, 10, 4)

    query_residuals = predictor(trend)

    assert query_residuals.shape == (5, 6, 4)


class HalfResidualPredictor(torch.nn.Module):
    def __init__(self, cfg: LambdaLabelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, trend: torch.Tensor) -> torch.Tensor:
        return torch.full((trend.shape[0], len(self.cfg.query_indices), trend.shape[2]), 0.5, device=trend.device)


def test_generate_lambda_labels_from_chunks_saves_requested_diagnostics():
    pytest.importorskip("scipy")
    cfg = LambdaLabelConfig(chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8))
    chunks = torch.zeros(2, 10, 1)
    chunks[:, cfg.query_indices] = 1.0
    indices = torch.tensor([100, 101])

    labels = generate_lambda_labels_from_chunks(
        chunks=chunks,
        indices=indices,
        predictor=HalfResidualPredictor(cfg),
        cfg=cfg,
        batch_size=1,
        save_diagnostics=True,
        max_diagnostic_chunks=1,
    )

    assert torch.equal(labels["valid_indices"], indices)
    assert torch.all(labels["lambda_t"] >= 0)
    assert torch.all(labels["lambda_t"] <= 1)
    assert labels["diagnostics"]["A_t"].shape == (1, 10, 1)
    assert torch.equal(labels["diagnostics"]["A_residual"][:, cfg.anchor_indices], torch.zeros(1, 4, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py -q
```

Expected: FAIL with import errors for `MaskedResidualPredictor` or `generate_lambda_labels_from_chunks`.

- [ ] **Step 3: Add residual predictor and generation helpers**

Append to `src/lerobot/policies/smolvla/lambda_labels.py`:

```python
from torch.utils.data import DataLoader, TensorDataset


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


@dataclass
class ResidualTrainingConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 256
    device: str = "cpu"


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
    device = next(predictor.parameters(), torch.empty(0)).device
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
        limit = chunks.shape[0] if max_diagnostic_chunks is None else min(max_diagnostic_chunks, chunks.shape[0])
        payload["diagnostics"] = {
            "A_t": chunks[:limit].detach().cpu(),
            "A_trend": trend[:limit].detach().cpu(),
            "A_residual": residuals[:limit].detach().cpu(),
            "A_hat": metrics.a_hat[:limit].detach().cpu(),
        }
    return payload
```

- [ ] **Step 4: Add runnable tool config and entrypoint**

Append to `src/lerobot/policies/smolvla/lambda_labels.py`:

```python
import argparse
from datetime import datetime, timezone


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SmolVLA residual predictor and generate lambda labels.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument("--max-diagnostic-chunks", type=int, default=128)
    return parser


def run_offline_lambda_label_generation(args: argparse.Namespace) -> None:
    from lerobot.configs.types import FeatureType

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, revision=args.revision)
    cfg = LambdaLabelConfig()
    chunk_batch = extract_action_chunks_from_dataset(dataset, cfg)
    action_stats = dataset.meta.stats[ACTION]
    action_norm_mode = NormalizationMode.MEAN_STD
    for feature_type, mode in getattr(dataset, "normalization_mapping", {}).items():
        if feature_type == FeatureType.ACTION:
            action_norm_mode = mode
    normalized_chunks = normalize_action_chunks(chunk_batch.chunks, action_stats, action_norm_mode)
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
        torch.save({"model_state_dict": predictor.state_dict(), "config": train_cfg.__dict__}, checkpoint_path)
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
        "created_at": datetime.now(timezone.utc).isoformat(),
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
```

Then adjust the normalization mode selection during implementation if the project has a policy config available in the CLI. The initial CLI can default to `MEAN_STD`, matching SmolVLA's default action normalization.

- [ ] **Step 5: Add console script**

Modify `pyproject.toml` under `[project.scripts]`:

```toml
lerobot-smolvla-lambda-labels="lerobot.policies.smolvla.lambda_labels:main"
```

- [ ] **Step 6: Run tests and CLI help**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py -q
uv run python -m lerobot.policies.smolvla.lambda_labels --help
```

Expected: tests PASS with SciPy-dependent tests skipped if SciPy is missing; help exits 0 and prints `Train SmolVLA residual predictor and generate lambda labels`.

- [ ] **Step 7: Commit**

```bash
git add src/lerobot/policies/smolvla/lambda_labels.py tests/policies/smolvla/test_lambda_labels.py pyproject.toml
git commit -m "feat: add smolvla lambda label generation tool"
```

---

### Task 4: Config and Preprocessor Label Injection

**Files:**
- Modify: `src/lerobot/policies/smolvla/configuration_smolvla.py`
- Modify: `src/lerobot/policies/smolvla/processor_smolvla.py`
- Modify: `tests/processor/test_smolvla_processor.py`

- [ ] **Step 1: Add failing processor test**

Append to `tests/processor/test_smolvla_processor.py`:

```python
def test_smolvla_preprocessor_injects_lambda_labels_from_sidecar(tmp_path):
    config = create_default_config()
    config.lambda_labels_path = str(tmp_path / "labels.pt")
    config.lambda_default_value = 0.1
    stats = create_default_stats()
    torch.save(
        {
            "valid_indices": torch.tensor([5]),
            "lambda_t": torch.tensor([0.35]),
            "metrics": {},
            "metadata": {},
        },
        config.lambda_labels_path,
    )

    with patch(
        "lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep", MockTokenizerProcessorStep
    ):
        preprocessor, _ = make_smolvla_pre_post_processors(config, stats)

    batch = {
        OBS_STATE: torch.randn(2, 8),
        OBS_IMAGE: torch.randn(2, 3, 224, 224),
        ACTION: torch.randn(2, 7),
        "task": ["pick", "place"],
        "index": torch.tensor([5, 6]),
    }
    processed = preprocessor(batch)

    assert torch.allclose(processed["lambda_t"], torch.tensor([0.35, 0.1]))
    assert torch.equal(processed["lambda_is_valid"], torch.tensor([True, False]))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/processor/test_smolvla_processor.py::test_smolvla_preprocessor_injects_lambda_labels_from_sidecar -q
```

Expected: FAIL because `SmolVLAConfig` has no `lambda_labels_path` field or processor does not inject labels.

- [ ] **Step 3: Add config fields and validation**

Modify `SmolVLAConfig` in `src/lerobot/policies/smolvla/configuration_smolvla.py`:

```python
    # Lambda residual-refinement labels and conditioning
    lambda_labels_path: str | None = None
    lambda_loss_weight: float = 0.05
    lambda_loss_type: str = "smooth_l1"
    lambda_conditioning: bool = True
    lambda_alpha_start: float = 0.0
    lambda_alpha_end: float = 1.0
    lambda_alpha_warmup_steps: int = 30_000
    lambda_default_value: float = 0.0

    # Dynamic closed-loop execution from predicted lambda
    dynamic_n_action_steps: bool = False
    dynamic_n_action_steps_min: int = 3
    dynamic_n_action_steps_max: int = 10
    lambda_ema_beta: float = 0.8
```

Extend `__post_init__`:

```python
        if self.lambda_loss_weight < 0:
            raise ValueError("lambda_loss_weight must be non-negative")
        if self.lambda_loss_type not in {"smooth_l1", "mse"}:
            raise ValueError("lambda_loss_type must be 'smooth_l1' or 'mse'")
        if not 0.0 <= self.lambda_alpha_start <= 1.0:
            raise ValueError("lambda_alpha_start must be in [0, 1]")
        if not 0.0 <= self.lambda_alpha_end <= 1.0:
            raise ValueError("lambda_alpha_end must be in [0, 1]")
        if self.lambda_alpha_warmup_steps < 0:
            raise ValueError("lambda_alpha_warmup_steps must be non-negative")
        if not 0.0 <= self.lambda_default_value <= 1.0:
            raise ValueError("lambda_default_value must be in [0, 1]")
        if self.dynamic_n_action_steps_min <= 0:
            raise ValueError("dynamic_n_action_steps_min must be positive")
        if self.dynamic_n_action_steps_max < self.dynamic_n_action_steps_min:
            raise ValueError("dynamic_n_action_steps_max must be >= dynamic_n_action_steps_min")
        if self.dynamic_n_action_steps_max > self.chunk_size:
            raise ValueError("dynamic_n_action_steps_max must be <= chunk_size")
        if not 0.0 <= self.lambda_ema_beta < 1.0:
            raise ValueError("lambda_ema_beta must be in [0, 1)")
```

- [ ] **Step 4: Add SmolVLA lambda label processor step**

Modify imports in `src/lerobot/policies/smolvla/processor_smolvla.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from lerobot.processor import ComplementaryDataProcessorStep, ProcessorStepRegistry
from .lambda_labels import LambdaLabelLookup, load_lambda_sidecar
```

Add this class above `make_smolvla_pre_post_processors`:

```python
@dataclass
@ProcessorStepRegistry.register(name="smolvla_lambda_label_processor")
class SmolVLALambdaLabelProcessorStep(ComplementaryDataProcessorStep):
    labels_path: str
    default_value: float = 0.0

    def __post_init__(self) -> None:
        self._lookup = LambdaLabelLookup(load_lambda_sidecar(Path(self.labels_path)), self.default_value)

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        if "index" not in complementary_data:
            complementary_data["lambda_t"] = torch.tensor(self.default_value, dtype=torch.float32)
            complementary_data["lambda_is_valid"] = torch.tensor(False, dtype=torch.bool)
            return complementary_data
        index = torch.as_tensor(complementary_data["index"], dtype=torch.long)
        lambda_t, lambda_is_valid = self._lookup.lookup(index)
        complementary_data["lambda_t"] = lambda_t
        complementary_data["lambda_is_valid"] = lambda_is_valid
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {"labels_path": self.labels_path, "default_value": self.default_value}

    def transform_features(self, features):
        return features
```

Insert the step into `input_steps` after `NewLineTaskProcessorStep()` and before `TokenizerProcessorStep(...)`:

```python
    if config.lambda_labels_path is not None:
        input_steps.append(
            SmolVLALambdaLabelProcessorStep(
                labels_path=config.lambda_labels_path,
                default_value=config.lambda_default_value,
            )
        )
```

Keep existing processors in the same relative order for the no-label case.

- [ ] **Step 5: Run processor tests**

Run:

```bash
uv run pytest tests/processor/test_smolvla_processor.py::test_make_smolvla_processor_basic tests/processor/test_smolvla_processor.py::test_smolvla_preprocessor_injects_lambda_labels_from_sidecar -q
```

Expected: PASS. The existing basic processor test still sees six steps when labels are disabled.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/processor_smolvla.py tests/processor/test_smolvla_processor.py
git commit -m "feat: inject smolvla lambda labels in preprocessing"
```

---

### Task 5: Lambda Token and Prediction Head in VLAFlowMatching

**Files:**
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Create: `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`

- [ ] **Step 1: Add failing tests using a fake VLM**

Create `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`:

```python
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import compute_dynamic_n_action_steps
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE


class FakeTokenizer:
    fake_image_token_id = 1
    global_image_token_id = 2


class FakeProcessor:
    tokenizer = FakeTokenizer()


class FakeSmolVLMWithExpertModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.expert_hidden_size = 8
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8, head_dim=4))
        self.processor = FakeProcessor()
        self.vlm = SimpleNamespace(device=torch.device("cpu"))

    def embed_language_tokens(self, tokens):
        return torch.zeros(tokens.shape[0], 8)

    def embed_image(self, image):
        return torch.zeros(image.shape[0], 2, 8)

    def forward(self, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, fill_kv_cache):
        outputs = []
        for item in inputs_embeds:
            outputs.append(item)
        return outputs, past_key_values


def make_config():
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        max_action_dim=3,
        max_state_dim=3,
        lambda_conditioning=True,
        dynamic_n_action_steps_min=1,
        dynamic_n_action_steps_max=4,
    )
    cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))}
    cfg.device = "cpu"
    return cfg


def test_embed_suffix_adds_lambda_token_before_action_tokens():
    with patch("lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel", FakeSmolVLMWithExpertModel):
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

        model = VLAFlowMatching(make_config())
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(
        noisy_actions=torch.zeros(2, 4, 3),
        timestep=torch.ones(2),
        lambda_cond=torch.tensor([0.2, 0.8]),
    )

    assert suffix_embs.shape == (2, 5, 8)
    assert suffix_pad_masks.shape == (2, 5)
    assert suffix_att_masks.shape == (2, 5)


def test_compute_lambda_condition_mixes_labels_and_stopped_prediction():
    with patch("lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel", FakeSmolVLMWithExpertModel):
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

        model = VLAFlowMatching(make_config())
    lambda_hat = torch.tensor([0.2, 0.8], requires_grad=True)
    lambda_t = torch.tensor([1.0, 0.0])
    valid = torch.tensor([True, False])

    cond = model.compute_lambda_condition(lambda_hat, lambda_t, valid, alpha=0.25)

    assert torch.allclose(cond, torch.tensor([0.8, 0.8]))
    assert cond.requires_grad is False


def test_compute_dynamic_n_action_steps_clamps_rounding():
    assert compute_dynamic_n_action_steps(0.0, n_min=3, n_max=10) == 10
    assert compute_dynamic_n_action_steps(1.0, n_min=3, n_max=10) == 3
    assert compute_dynamic_n_action_steps(0.5, n_min=3, n_max=10) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py -q
```

Expected: FAIL because `compute_dynamic_n_action_steps`, lambda token support, or `compute_lambda_condition` does not exist.

- [ ] **Step 3: Add lambda head, token MLP, and helper functions**

Modify `VLAFlowMatching.__init__` in `src/lerobot/policies/smolvla/modeling_smolvla.py` after `action_time_mlp_out`:

```python
        self.lambda_head = nn.Sequential(
            nn.Linear(self.vlm_with_expert.config.text_config.hidden_size, self.vlm_with_expert.config.text_config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.vlm_with_expert.config.text_config.hidden_size, 1),
        )
        self.lambda_token_mlp = nn.Sequential(
            nn.Linear(1, self.vlm_with_expert.expert_hidden_size),
            nn.SiLU(),
            nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size),
        )
```

Add module-level helper near `ActionSelectKwargs`:

```python
def compute_dynamic_n_action_steps(lambda_value: float, n_min: int, n_max: int) -> int:
    lambda_value = max(0.0, min(1.0, float(lambda_value)))
    return max(n_min, min(n_max, round(n_max - lambda_value * (n_max - n_min))))
```

Add methods to `VLAFlowMatching`:

```python
    def predict_lambda_from_prefix(self, prefix_out: Tensor, prefix_pad_masks: Tensor) -> Tensor:
        mask = prefix_pad_masks.to(dtype=prefix_out.dtype).unsqueeze(-1)
        pooled = (prefix_out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return torch.sigmoid(self.lambda_head(pooled)).squeeze(-1)

    def compute_lambda_condition(
        self,
        lambda_hat: Tensor,
        lambda_t: Tensor | None,
        lambda_is_valid: Tensor | None,
        alpha: float,
    ) -> Tensor:
        pred = lambda_hat.detach()
        if lambda_t is None or lambda_is_valid is None:
            return pred
        labels = lambda_t.to(device=lambda_hat.device, dtype=lambda_hat.dtype)
        valid = lambda_is_valid.to(device=lambda_hat.device, dtype=torch.bool)
        mixed = (1.0 - alpha) * labels + alpha * pred
        return torch.where(valid, mixed, pred).detach()
```

Change `embed_suffix` signature and body:

```python
    def embed_suffix(self, noisy_actions, timestep, lambda_cond: Tensor | None = None):
```

Insert before action tokens are appended:

```python
        if self.config.lambda_conditioning and lambda_cond is not None:
            lambda_token = self.lambda_token_mlp(lambda_cond[:, None].to(dtype=dtype))[:, None, :]
            embs.append(lambda_token)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]
```

Change the suffix attention mask action line:

```python
        att_masks += [1] * self.config.chunk_size
```

The existing line remains valid because the lambda token adds its own mask before it.

- [ ] **Step 4: Thread lambda token through denoise step**

Change `denoise_step` signature:

```python
        lambda_cond: Tensor | None = None,
```

Change the suffix call:

```python
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep, lambda_cond=lambda_cond)
```

Keep:

```python
        suffix_out = suffix_out[:, -self.config.chunk_size :]
```

This slices away the lambda token and keeps only action-token outputs.

- [ ] **Step 5: Run lambda conditioning tests**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py
git commit -m "feat: add smolvla lambda token conditioning"
```

---

### Task 6: Lambda Auxiliary Loss and Dynamic Inference

**Files:**
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Modify: `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`

- [ ] **Step 1: Add failing unit tests for alpha schedule and EMA execution**

Append to `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`:

```python
def test_lambda_alpha_schedule_reaches_end_value():
    with patch("lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel", FakeSmolVLMWithExpertModel):
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

        cfg = make_config()
        cfg.lambda_alpha_start = 0.0
        cfg.lambda_alpha_end = 1.0
        cfg.lambda_alpha_warmup_steps = 10
        model = VLAFlowMatching(cfg)

    model.lambda_train_step.fill_(5)
    assert model.compute_lambda_alpha() == 0.5
    model.lambda_train_step.fill_(10)
    assert model.compute_lambda_alpha() == 1.0


def test_lambda_ema_updates_execution_horizon_on_policy_shell():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = object.__new__(SmolVLAPolicy)
    policy.config = make_config()
    policy.config.dynamic_n_action_steps = True
    policy.config.lambda_ema_beta = 0.5
    policy._lambda_smooth = None

    assert policy._update_dynamic_n_action_steps(torch.tensor([1.0])) == 1
    assert policy._lambda_smooth == 1.0
    assert policy._update_dynamic_n_action_steps(torch.tensor([0.0])) == 2
    assert policy._lambda_smooth == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py -q
```

Expected: FAIL because `lambda_train_step`, `compute_lambda_alpha`, or `_update_dynamic_n_action_steps` does not exist.

- [ ] **Step 3: Add train-step buffer and alpha schedule**

In `VLAFlowMatching.__init__`, add:

```python
        self.register_buffer("lambda_train_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.last_lambda_hat: Tensor | None = None
```

Add methods:

```python
    def compute_lambda_alpha(self) -> float:
        if self.config.lambda_alpha_warmup_steps == 0:
            return float(self.config.lambda_alpha_end)
        progress = min(1.0, float(self.lambda_train_step.item()) / float(self.config.lambda_alpha_warmup_steps))
        return float(self.config.lambda_alpha_start + progress * (self.config.lambda_alpha_end - self.config.lambda_alpha_start))

    def update_lambda_train_step(self) -> None:
        self.lambda_train_step.add_(1)
```

In `SmolVLAPolicy`, add:

```python
    def update(self):
        if hasattr(self.model, "update_lambda_train_step"):
            self.model.update_lambda_train_step()
```

- [ ] **Step 4: Update VLAFlowMatching forward and sample_actions**

Change `VLAFlowMatching.forward` signature:

```python
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None, lambda_t=None, lambda_is_valid=None
```

Use prefix-cache path when lambda conditioning is enabled:

```python
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        if self.config.lambda_conditioning:
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_outputs, past_key_values = self.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
                fill_kv_cache=True,
            )
            lambda_hat = self.predict_lambda_from_prefix(prefix_outputs[0].to(dtype=torch.float32), prefix_pad_masks)
            lambda_cond = self.compute_lambda_condition(
                lambda_hat=lambda_hat,
                lambda_t=lambda_t,
                lambda_is_valid=lambda_is_valid,
                alpha=self.compute_lambda_alpha(),
            )
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time,
                lambda_cond=lambda_cond,
            )
            losses = F.mse_loss(u_t, v_t, reduction="none")
            self.last_lambda_hat = lambda_hat.detach()
            return {"losses": losses, "lambda_hat": lambda_hat}
```

Keep the existing full forward path under an `else` branch and return:

```python
        return {"losses": losses, "lambda_hat": None}
```

In `sample_actions`, after prefix cache creation:

```python
        lambda_hat = None
        lambda_cond = None
        if self.config.lambda_conditioning:
            lambda_hat = self.predict_lambda_from_prefix(_[0].to(dtype=torch.float32), prefix_pad_masks)
            lambda_cond = lambda_hat.detach()
            self.last_lambda_hat = lambda_cond
```

Rename `_` to `prefix_outputs` in that block so this reads clearly:

```python
        prefix_outputs, past_key_values = self.vlm_with_expert.forward(...)
```

Pass `lambda_cond=lambda_cond` into every `denoise_step` call.

- [ ] **Step 5: Update SmolVLAPolicy.forward for lambda auxiliary loss**

In `SmolVLAPolicy.forward`, change model call:

```python
        model_output = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            lambda_t=batch.get("lambda_t"),
            lambda_is_valid=batch.get("lambda_is_valid"),
        )
        losses = model_output["losses"]
        lambda_hat = model_output.get("lambda_hat")
```

After flow per-sample loss is computed, add lambda auxiliary handling:

```python
        lambda_loss_per_sample = None
        lambda_t = batch.get("lambda_t")
        lambda_is_valid = batch.get("lambda_is_valid")
        if lambda_hat is not None and lambda_t is not None and lambda_is_valid is not None:
            lambda_t = lambda_t.to(device=lambda_hat.device, dtype=lambda_hat.dtype).view_as(lambda_hat)
            lambda_is_valid = lambda_is_valid.to(device=lambda_hat.device, dtype=torch.bool).view_as(lambda_hat)
            if self.config.lambda_loss_type == "smooth_l1":
                raw_lambda_loss = F.smooth_l1_loss(lambda_hat, lambda_t, reduction="none")
            else:
                raw_lambda_loss = F.mse_loss(lambda_hat, lambda_t, reduction="none")
            lambda_loss_per_sample = torch.where(lambda_is_valid, raw_lambda_loss, torch.zeros_like(raw_lambda_loss))
            valid_count = lambda_is_valid.sum().clamp_min(1)
            lambda_loss = lambda_loss_per_sample.sum() / valid_count
            loss_dict["lambda_loss"] = lambda_loss.item()
            loss_dict["lambda_hat_mean"] = lambda_hat.detach().mean().item()
            loss_dict["lambda_valid_frac"] = lambda_is_valid.float().mean().item()
```

For `reduction == "none"`:

```python
            if lambda_loss_per_sample is not None:
                per_sample_loss = per_sample_loss + self.config.lambda_loss_weight * lambda_loss_per_sample
```

For scalar reduction:

```python
            if lambda_loss_per_sample is not None:
                loss = loss + self.config.lambda_loss_weight * lambda_loss
```

Set `loss_dict["loss"]` after adding auxiliary loss.

- [ ] **Step 6: Add dynamic execution helper in SmolVLAPolicy**

In `reset`, add:

```python
        self._lambda_smooth = None
```

Add method:

```python
    def _update_dynamic_n_action_steps(self, lambda_hat: Tensor | None) -> int:
        if not self.config.dynamic_n_action_steps or lambda_hat is None:
            return self.config.n_action_steps
        lambda_now = float(lambda_hat.detach().float().mean().clamp(0.0, 1.0).cpu().item())
        if self._lambda_smooth is None:
            self._lambda_smooth = lambda_now
        else:
            beta = self.config.lambda_ema_beta
            self._lambda_smooth = beta * self._lambda_smooth + (1.0 - beta) * lambda_now
        return compute_dynamic_n_action_steps(
            self._lambda_smooth,
            n_min=self.config.dynamic_n_action_steps_min,
            n_max=min(self.config.dynamic_n_action_steps_max, self.config.chunk_size),
        )
```

In `select_action`, change the queue extension block:

```python
            n_exec = self._update_dynamic_n_action_steps(getattr(self.model, "last_lambda_hat", None))
            self._queues[ACTION].extend(actions.transpose(0, 1)[:n_exec])
```

- [ ] **Step 7: Run targeted tests**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/policies/smolvla/test_lambda_labels.py tests/processor/test_smolvla_processor.py::test_smolvla_preprocessor_injects_lambda_labels_from_sidecar -q
```

Expected: PASS, with SciPy-dependent tests skipped if SciPy is not installed.

- [ ] **Step 8: Commit**

```bash
git add src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py
git commit -m "feat: train smolvla with lambda auxiliary loss"
```

---

### Task 7: Final Verification and Cleanup

**Files:**
- Review: `src/lerobot/policies/smolvla/lambda_labels.py`
- Review: `src/lerobot/policies/smolvla/configuration_smolvla.py`
- Review: `src/lerobot/policies/smolvla/processor_smolvla.py`
- Review: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Review: `tests/policies/smolvla/test_lambda_labels.py`
- Review: `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`
- Review: `tests/processor/test_smolvla_processor.py`
- Review: `pyproject.toml`

- [ ] **Step 1: Run SmolVLA and processor tests**

Run:

```bash
uv run pytest tests/policies/smolvla tests/processor/test_smolvla_processor.py -q
```

Expected: PASS, with CUDA/transformers/pretrained-weight tests skipped by existing decorators when unavailable.

- [ ] **Step 2: Run lint/format checks for touched files**

Run:

```bash
uv run ruff check src/lerobot/policies/smolvla/lambda_labels.py src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/processor_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_lambda_labels.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/processor/test_smolvla_processor.py
```

Expected: PASS. If formatting is reported, run:

```bash
uv run ruff format src/lerobot/policies/smolvla/lambda_labels.py src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/processor_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_lambda_labels.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/processor/test_smolvla_processor.py
uv run ruff check src/lerobot/policies/smolvla/lambda_labels.py src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/processor_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_lambda_labels.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/processor/test_smolvla_processor.py
```

Expected after formatting: PASS.

- [ ] **Step 3: Verify offline CLI help**

Run:

```bash
uv run lerobot-smolvla-lambda-labels --help
```

Expected: exits 0 and prints arguments including `--repo-id`, `--output-path`, `--epochs`, and `--save-diagnostics`.

- [ ] **Step 4: Check git diff**

Run:

```bash
git status --short
git diff --check
```

Expected: only intentional files are modified; `git diff --check` prints no whitespace errors.

- [ ] **Step 5: Commit final cleanup if needed**

If Step 2 formatting changed files:

```bash
git add src/lerobot/policies/smolvla/lambda_labels.py src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/processor_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_lambda_labels.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/processor/test_smolvla_processor.py pyproject.toml
git commit -m "chore: format smolvla lambda label changes"
```

If no formatting changed files, do not create an empty commit.

---

## Self-Review Notes

- Spec coverage: offline normalized action chunks, PCHIP anchors, masked residual query prediction, lambda metrics, sidecar labels, single-start assignment, SmolVLA lambda auxiliary loss, suffix token conditioning, alpha schedule, and dynamic inference are covered by Tasks 1 through 6.
- Red-flag scan: this plan uses concrete paths, commands, tests, and code blocks. There are no deferred sections.
- Type consistency: the same names are used across tasks: `LambdaLabelConfig`, `LambdaMetrics`, `MaskedResidualPredictor`, `LambdaLabelLookup`, `SmolVLALambdaLabelProcessorStep`, `lambda_t`, `lambda_is_valid`, `lambda_hat`, `lambda_cond`, and `compute_dynamic_n_action_steps`.
