import pytest
import torch

from lerobot.configs.types import NormalizationMode
from lerobot.policies.smolvla.lambda_labels import (
    LambdaLabelConfig,
    LambdaLabelLookup,
    MaskedResidualPredictor,
    compute_lambda_metrics,
    compute_pchip_trend,
    generate_lambda_labels_from_chunks,
    load_lambda_sidecar,
    normalize_action_chunks,
    save_lambda_sidecar,
    scatter_query_residuals,
)


def test_compute_pchip_trend_preserves_anchor_values():
    pytest.importorskip("scipy")
    cfg = LambdaLabelConfig(
        chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8)
    )
    chunks = torch.zeros(2, 10, 3)
    chunks[:, 0] = torch.tensor([0.0, 1.0, 2.0])
    chunks[:, 3] = torch.tensor([3.0, 4.0, 5.0])
    chunks[:, 6] = torch.tensor([6.0, 7.0, 8.0])
    chunks[:, 9] = torch.tensor([9.0, 10.0, 11.0])

    trend = compute_pchip_trend(chunks, cfg)

    assert trend.shape == chunks.shape
    assert torch.allclose(trend[:, cfg.anchor_indices], chunks[:, cfg.anchor_indices], atol=1e-6)


def test_scatter_query_residuals_keeps_anchor_residual_zero():
    cfg = LambdaLabelConfig(
        chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8)
    )
    query_residuals = torch.ones(4, len(cfg.query_indices), 2)

    full_residuals = scatter_query_residuals(query_residuals, action_dim=2, cfg=cfg)

    assert full_residuals.shape == (4, 10, 2)
    assert torch.equal(full_residuals[:, cfg.anchor_indices], torch.zeros(4, len(cfg.anchor_indices), 2))
    assert torch.equal(full_residuals[:, cfg.query_indices], torch.ones(4, len(cfg.query_indices), 2))


def test_compute_lambda_metrics_uses_query_positions_and_clamps():
    cfg = LambdaLabelConfig(
        chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8)
    )
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
    cfg = LambdaLabelConfig(
        chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8)
    )
    chunks = torch.zeros(1, 10, 1)
    trend = torch.zeros_like(chunks)
    residuals = torch.zeros_like(chunks)
    chunks[:, cfg.query_indices] = 1.0
    residuals[:, cfg.query_indices] = -2.0

    metrics = compute_lambda_metrics(chunks, trend, residuals, cfg)

    assert metrics.e_full.item() > metrics.e_trend.item()
    assert metrics.lambda_t.item() == 0.0


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


def test_masked_residual_predictor_outputs_query_residual_shape():
    cfg = LambdaLabelConfig(
        chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8)
    )
    predictor = MaskedResidualPredictor(
        chunk_size=10, action_dim=4, num_query_positions=len(cfg.query_indices), hidden_dim=16
    )
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
    cfg = LambdaLabelConfig(
        chunk_size=10, anchor_indices=(0, 3, 6, 9), query_indices=(1, 2, 4, 5, 7, 8)
    )
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
