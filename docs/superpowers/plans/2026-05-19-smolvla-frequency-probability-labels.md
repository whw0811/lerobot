# SmolVLA Frequency Probability Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scalar SmolVLA lambda labels with 3-way DCT frequency probability labels and use non-low-frequency probability to gate adapter corrections.

**Architecture:** The offline label path in `lambda_labels.py` computes `[p_low, p_mid, p_high]` from 10-step DCT band energy and stores it in the existing `lambda_t` sidecar field. The processor keeps exact-index lookup but returns `[B, 3]` labels. The model predicts 3 logits from VLM prefix hidden states, supervises them with soft-label cross entropy, and gates the velocity adapter with `p_mid + p_high`.

**Tech Stack:** Python 3.12, PyTorch, pytest, uv, ruff, existing SmolVLA policy modules.

---

## File Structure

- Modify `src/lerobot/policies/smolvla/lambda_labels.py`
  - Add DCT frequency probability helpers.
  - Change generated `lambda_t` to `[N, 3]`.
  - Keep old scalar sidecars readable by mapping scalar `lambda` to `[1 - lambda, 0, lambda]`.
- Modify `src/lerobot/policies/smolvla/processor_smolvla.py`
  - Keep the same processor API but inject `[B, 3]` `lambda_t`.
  - Use `[1, 0, 0]` for missing labels.
- Modify `src/lerobot/policies/smolvla/modeling_smolvla.py`
  - Predict lambda logits and probabilities.
  - Replace scalar lambda loss with soft-label cross entropy.
  - Change adapter conditioning from scalar to 3-way distribution.
  - Gate adapter strength with `1 - p_low`.
  - Derive dynamic execution urgency from `p_mid + p_high`.
- Modify `src/lerobot/policies/smolvla/configuration_smolvla.py`
  - Add `lambda_default_distribution` config and validate it.
- Modify `tests/policies/smolvla/test_lambda_labels.py`
  - Add DCT distribution and sidecar migration tests.
- Modify `tests/processor/test_smolvla_processor.py`
  - Update sidecar injection test for `[B, 3]`.
- Modify `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`
  - Update lightweight model tests for 3-way labels, logits loss, gate, and dynamic urgency.

---

### Task 1: DCT Frequency Probability Labels

**Files:**
- Modify: `src/lerobot/policies/smolvla/lambda_labels.py`
- Modify: `tests/policies/smolvla/test_lambda_labels.py`

- [ ] **Step 1: Write failing tests for DCT band probability labels**

Append these imports and tests to `tests/policies/smolvla/test_lambda_labels.py`:

```python
from lerobot.policies.smolvla.lambda_labels import (
    compute_dct_frequency_probabilities,
)


def _cosine_signal(k: int, chunk_size: int = 10) -> torch.Tensor:
    n = torch.arange(chunk_size, dtype=torch.float32)
    return torch.cos(torch.pi / chunk_size * (n + 0.5) * k).view(1, chunk_size, 1)


def test_compute_dct_frequency_probabilities_detects_low_mid_high_bands():
    low = compute_dct_frequency_probabilities(_cosine_signal(1))
    mid = compute_dct_frequency_probabilities(_cosine_signal(4))
    high = compute_dct_frequency_probabilities(_cosine_signal(8))

    assert low.argmax(dim=-1).item() == 0
    assert mid.argmax(dim=-1).item() == 1
    assert high.argmax(dim=-1).item() == 2
    torch.testing.assert_close(low.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(mid.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(high.sum(dim=-1), torch.ones(1))


def test_compute_dct_frequency_probabilities_flat_chunk_defaults_to_low():
    probs = compute_dct_frequency_probabilities(torch.zeros(2, 10, 3))

    torch.testing.assert_close(probs, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
```

- [ ] **Step 2: Run the new tests to verify RED**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py::test_compute_dct_frequency_probabilities_detects_low_mid_high_bands tests/policies/smolvla/test_lambda_labels.py::test_compute_dct_frequency_probabilities_flat_chunk_defaults_to_low -q
```

Expected result: FAIL with `ImportError` or `AttributeError` because `compute_dct_frequency_probabilities` does not exist.

- [ ] **Step 3: Implement the DCT probability helper**

Add this near the existing DCT helper in `src/lerobot/policies/smolvla/lambda_labels.py`:

```python
DEFAULT_DCT_FREQUENCY_BANDS: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4, 5), (6, 7, 8, 9))


def compute_dct_coefficients(action_chunks: Tensor) -> Tensor:
    if action_chunks.ndim != 3:
        raise ValueError("Expected action_chunks shaped [N, T, D]")
    chunk_size = action_chunks.shape[1]
    chunks = action_chunks.float()
    n = torch.arange(chunk_size, dtype=chunks.dtype, device=chunks.device)
    k = torch.arange(chunk_size, dtype=chunks.dtype, device=chunks.device)
    basis = torch.cos(torch.pi / chunk_size * (n[None, :] + 0.5) * k[:, None])
    return torch.einsum("kt,btd->bkd", basis, chunks)


def compute_dct_frequency_probabilities(
    action_chunks: Tensor,
    bands: tuple[tuple[int, ...], ...] = DEFAULT_DCT_FREQUENCY_BANDS,
    eps: float = 1e-6,
) -> Tensor:
    coeffs = compute_dct_coefficients(action_chunks)
    chunk_size = coeffs.shape[1]
    if any(any(index < 0 or index >= chunk_size for index in band) for band in bands):
        raise ValueError("DCT frequency band indices must be inside chunk_size")
    if sorted(index for band in bands for index in band) != list(range(chunk_size)):
        raise ValueError("DCT frequency bands must cover every coefficient exactly once")

    coeff_energy = coeffs.pow(2).sum(dim=2)
    band_energies = torch.stack(
        [coeff_energy[:, torch.tensor(band, dtype=torch.long, device=coeff_energy.device)].sum(dim=1) for band in bands],
        dim=1,
    )
    total_energy = band_energies.sum(dim=1, keepdim=True)
    probs = band_energies / total_energy.clamp_min(eps)
    low_default = torch.zeros_like(probs)
    low_default[:, 0] = 1.0
    return torch.where(total_energy > eps, probs, low_default).to(dtype=action_chunks.dtype)
```

Update `compute_dct_high_frequency_ratio` to reuse `compute_dct_coefficients` while preserving its centered high-frequency behavior:

```python
centered = action_chunks.float() - action_chunks.float().mean(dim=1, keepdim=True)
coeffs = compute_dct_coefficients(centered)
```

- [ ] **Step 4: Run the DCT tests to verify GREEN**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py::test_compute_dct_frequency_probabilities_detects_low_mid_high_bands tests/policies/smolvla/test_lambda_labels.py::test_compute_dct_frequency_probabilities_flat_chunk_defaults_to_low -q
```

Expected result: PASS.

- [ ] **Step 5: Write failing test for generated label shape and probability sum**

Change `test_generate_lambda_labels_from_chunks_saves_requested_diagnostics` in `tests/policies/smolvla/test_lambda_labels.py` so the label assertions read:

```python
    assert labels["lambda_t"].shape == (2, 3)
    assert torch.all(labels["lambda_t"] >= 0)
    torch.testing.assert_close(labels["lambda_t"].sum(dim=-1), torch.ones(2))
```

- [ ] **Step 6: Run the generation test to verify RED**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py::test_generate_lambda_labels_from_chunks_saves_requested_diagnostics -q
```

Expected result: FAIL because `lambda_t` is still scalar shaped `[2]`.

- [ ] **Step 7: Make generated `lambda_t` use DCT frequency probabilities**

In `generate_lambda_labels_from_chunks`, compute `lambda_frequency = compute_dct_frequency_probabilities(chunks, eps=cfg.eps)` after the existing PCHIP metrics are computed. Return it as the canonical `lambda_t`:

```python
    lambda_frequency = compute_dct_frequency_probabilities(chunks, eps=cfg.eps)
```

Change the return dictionary entries to:

```python
        "lambda_pchip": metrics.lambda_t.detach().cpu().to(dtype=torch.float32),
        "lambda_t": lambda_frequency.detach().cpu().to(dtype=torch.float32),
        "alpha_conf": alpha_conf.detach().cpu().to(dtype=torch.float32),
        "dct_frequency_probs": lambda_frequency.detach().cpu().to(dtype=torch.float32),
```

Keep `lambda_pchip`, `lambda_raw`, `lambda_envelope`, `dct_high_freq_ratio`, and `lambda_dct` as diagnostics so existing analysis tooling still has the old signals.

- [ ] **Step 8: Run the generation test to verify GREEN**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py::test_generate_lambda_labels_from_chunks_saves_requested_diagnostics -q
```

Expected result: PASS.

- [ ] **Step 9: Commit Task 1**

Run:

```bash
git add src/lerobot/policies/smolvla/lambda_labels.py tests/policies/smolvla/test_lambda_labels.py
git commit -m "feat: generate smolvla frequency probability labels"
```

---

### Task 2: Sidecar Lookup And Processor Distribution Injection

**Files:**
- Modify: `src/lerobot/policies/smolvla/lambda_labels.py`
- Modify: `src/lerobot/policies/smolvla/processor_smolvla.py`
- Modify: `src/lerobot/policies/smolvla/configuration_smolvla.py`
- Modify: `tests/policies/smolvla/test_lambda_labels.py`
- Modify: `tests/processor/test_smolvla_processor.py`

- [ ] **Step 1: Write failing lookup tests for distribution labels and scalar migration**

Append to `tests/policies/smolvla/test_lambda_labels.py`:

```python
def test_lambda_sidecar_lookup_returns_distribution_and_low_default(tmp_path):
    path = tmp_path / "labels.pt"
    save_lambda_sidecar(
        path=path,
        valid_indices=torch.tensor([10, 12]),
        lambda_t=torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]]),
        metrics={},
        metadata={"repo_id": "unit/test"},
    )

    lookup = LambdaLabelLookup(load_lambda_sidecar(path), default_value=(1.0, 0.0, 0.0))
    values, confidence, valid = lookup.lookup_with_confidence(torch.tensor([9, 10, 12]))

    torch.testing.assert_close(values, torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.7, 0.2, 0.1]]))
    torch.testing.assert_close(confidence, torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(valid, torch.tensor([False, True, True]))


def test_lambda_sidecar_lookup_maps_legacy_scalar_to_low_high_distribution(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "valid_indices": torch.tensor([5]),
            "lambda_t": torch.tensor([0.25]),
            "metrics": {},
            "metadata": {},
        },
        path,
    )

    lookup = LambdaLabelLookup(load_lambda_sidecar(path), default_value=(1.0, 0.0, 0.0))
    values, valid = lookup.lookup(torch.tensor([5]))

    torch.testing.assert_close(values, torch.tensor([[0.75, 0.0, 0.25]]))
    assert torch.equal(valid, torch.tensor([True]))
```

- [ ] **Step 2: Run lookup tests to verify RED**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py::test_lambda_sidecar_lookup_returns_distribution_and_low_default tests/policies/smolvla/test_lambda_labels.py::test_lambda_sidecar_lookup_maps_legacy_scalar_to_low_high_distribution -q
```

Expected result: FAIL because `save_lambda_sidecar` and `LambdaLabelLookup` still flatten labels.

- [ ] **Step 3: Update sidecar save and lookup for `[N, 3]` labels**

In `src/lerobot/policies/smolvla/lambda_labels.py`, add:

```python
def _as_lambda_distribution(values: Tensor, *, default_if_empty: bool = False) -> Tensor:
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.ndim == 0:
        values = values.view(1)
    if values.ndim == 1:
        values = values.clamp(0.0, 1.0)
        return torch.stack([1.0 - values, torch.zeros_like(values), values], dim=-1)
    if values.ndim == 2 and values.shape[1] == 3:
        row_sum = values.sum(dim=-1, keepdim=True)
        low_default = torch.zeros_like(values)
        low_default[:, 0] = 1.0
        normalized = values.clamp_min(0.0) / row_sum.clamp_min(1e-6)
        return torch.where(row_sum > 1e-6, normalized, low_default)
    if default_if_empty and values.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32)
    raise ValueError("lambda labels must be shaped [N] for legacy scalar labels or [N, 3] for frequency probabilities")
```

Change `save_lambda_sidecar` to keep `lambda_t` as a distribution:

```python
    lambda_t = _as_lambda_distribution(lambda_t)
    lambda_pchip = torch.as_tensor(lambda_pchip, dtype=torch.float32).view(-1)
```

Update the matching-length check:

```python
    if not (valid_indices.numel() == lambda_pchip.numel() == lambda_t.shape[0] == alpha_conf.numel()):
        raise ValueError("valid_indices, lambda_pchip, lambda_t, and alpha_conf must have matching lengths")
```

In `LambdaLabelLookup.__init__`, change value handling:

```python
        raw_values = torch.as_tensor(payload.get("lambda_t", payload.get("lambda_pchip")), dtype=torch.float32)
        values = _as_lambda_distribution(raw_values, default_if_empty=True)
```

Change default initialization in `lookup_with_confidence`:

```python
        default = torch.as_tensor(self.default_value, dtype=torch.float32)
        if default.ndim == 0:
            default = _as_lambda_distribution(default.view(1))[0]
        if default.shape != (3,):
            raise ValueError("default lambda distribution must have shape [3]")
        values = default.view(1, 3).expand(flat_indices.numel(), 3).clone()
```

Keep confidence and valid as 1D tensors.

- [ ] **Step 4: Run lookup tests to verify GREEN**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py::test_lambda_sidecar_lookup_returns_distribution_and_low_default tests/policies/smolvla/test_lambda_labels.py::test_lambda_sidecar_lookup_maps_legacy_scalar_to_low_high_distribution -q
```

Expected result: PASS.

- [ ] **Step 5: Write failing processor/config test for default distribution injection**

Modify `test_smolvla_preprocessor_injects_lambda_labels_from_sidecar` in `tests/processor/test_smolvla_processor.py`:

```python
    config.lambda_default_distribution = (1.0, 0.0, 0.0)
```

Use this sidecar payload:

```python
            "lambda_t": torch.tensor([[0.2, 0.3, 0.5]]),
```

Change assertions:

```python
    torch.testing.assert_close(processed["lambda_t"], torch.tensor([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]]))
    torch.testing.assert_close(processed["lambda_confidence"], torch.tensor([1.0, 0.0]))
    assert torch.equal(processed["lambda_is_valid"], torch.tensor([True, False]))
```

- [ ] **Step 6: Run processor test to verify RED**

Run:

```bash
uv run pytest tests/processor/test_smolvla_processor.py::test_smolvla_preprocessor_injects_lambda_labels_from_sidecar -q
```

Expected result: FAIL because `SmolVLAConfig` has no `lambda_default_distribution` and the processor uses scalar default value.

- [ ] **Step 7: Add config and processor distribution defaults**

In `SmolVLAConfig`, add:

```python
    lambda_default_distribution: tuple[float, float, float] = (1.0, 0.0, 0.0)
```

In `__post_init__`, validate:

```python
        if len(self.lambda_default_distribution) != 3:
            raise ValueError("lambda_default_distribution must contain three probabilities")
        if any(value < 0 for value in self.lambda_default_distribution):
            raise ValueError("lambda_default_distribution must be non-negative")
        default_sum = sum(self.lambda_default_distribution)
        if default_sum <= 0:
            raise ValueError("lambda_default_distribution must have positive sum")
        self.lambda_default_distribution = tuple(
            float(value) / float(default_sum) for value in self.lambda_default_distribution
        )
```

In `SmolVLALambdaLabelProcessorStep.__init__`, accept `default_value: float | tuple[float, float, float]`. In `make_smolvla_pre_post_processors`, pass:

```python
                default_value=config.lambda_default_distribution,
```

In `SmolVLALambdaLabelProcessorStep.__call__`, when `index` is missing, set:

```python
            complementary_data["lambda_t"] = torch.tensor(self.default_value, dtype=torch.float32)
```

- [ ] **Step 8: Run processor test to verify GREEN**

Run:

```bash
uv run pytest tests/processor/test_smolvla_processor.py::test_smolvla_preprocessor_injects_lambda_labels_from_sidecar -q
```

Expected result: PASS.

- [ ] **Step 9: Commit Task 2**

Run:

```bash
git add src/lerobot/policies/smolvla/lambda_labels.py src/lerobot/policies/smolvla/processor_smolvla.py src/lerobot/policies/smolvla/configuration_smolvla.py tests/policies/smolvla/test_lambda_labels.py tests/processor/test_smolvla_processor.py
git commit -m "feat: load smolvla frequency probability sidecars"
```

---

### Task 3: Soft-Label Loss, Distribution Condition, And Adapter Gate

**Files:**
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Modify: `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`

- [ ] **Step 1: Write failing tests for distribution loss and condition mixing**

In `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`, import:

```python
import torch.nn.functional as F
```

Replace `test_compute_lambda_condition_mixes_labels_and_stopped_prediction` with:

```python
def test_compute_lambda_condition_mixes_distribution_labels_and_stopped_prediction():
    with patch(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel", FakeSmolVLMWithExpertModel
    ):
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

        model = VLAFlowMatching(make_config())
    lambda_hat = torch.tensor([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]], requires_grad=True)
    lambda_t = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    valid = torch.tensor([True, False])

    cond = model.compute_lambda_condition(lambda_hat, lambda_t, valid, alpha=0.25)

    torch.testing.assert_close(cond, torch.tensor([[0.8, 0.075, 0.125], [0.8, 0.1, 0.1]]))
    assert cond.requires_grad is False
```

Add:

```python
def test_compute_lambda_supervision_loss_uses_soft_label_cross_entropy_and_confidence():
    from lerobot.policies.smolvla.modeling_smolvla import compute_lambda_supervision_loss

    logits = torch.tensor([[2.0, 0.0, -2.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.0, 0.0, 1.0]])
    valid = torch.tensor([True, True, False])
    confidence = torch.tensor([1.0, 0.5, 1.0])

    loss, per_sample, stats = compute_lambda_supervision_loss(
        lambda_logits=logits,
        lambda_t=labels,
        lambda_is_valid=valid,
        lambda_confidence=confidence,
    )

    raw = -(labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    expected = (raw[0] * 1.0 + raw[1] * 0.5) / 1.5
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(per_sample, torch.tensor([raw[0], raw[1] * 0.5, 0.0]))
    assert stats["lambda_confidence_mean"] == 0.75
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py::test_compute_lambda_condition_mixes_distribution_labels_and_stopped_prediction tests/policies/smolvla/test_smolvla_lambda_conditioning.py::test_compute_lambda_supervision_loss_uses_soft_label_cross_entropy_and_confidence -q
```

Expected result: FAIL because the implementation expects scalar labels and the loss signature still uses `lambda_hat`.

- [ ] **Step 3: Update loss and condition helpers**

In `src/lerobot/policies/smolvla/modeling_smolvla.py`, replace `compute_lambda_supervision_loss` with:

```python
def compute_lambda_supervision_loss(
    lambda_logits: Tensor,
    lambda_t: Tensor,
    lambda_is_valid: Tensor,
    lambda_confidence: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    labels = lambda_t.to(device=lambda_logits.device, dtype=lambda_logits.dtype)
    if labels.ndim == 1:
        labels = torch.stack([1.0 - labels, torch.zeros_like(labels), labels], dim=-1)
    labels = labels.view(lambda_logits.shape)
    labels = labels.clamp_min(0.0)
    labels = labels / labels.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    valid = lambda_is_valid.to(device=lambda_logits.device, dtype=torch.bool).view(lambda_logits.shape[0])
    if lambda_confidence is None:
        confidence = torch.ones(lambda_logits.shape[0], device=lambda_logits.device, dtype=lambda_logits.dtype)
    else:
        confidence = lambda_confidence.to(device=lambda_logits.device, dtype=lambda_logits.dtype).view(lambda_logits.shape[0])
        confidence = confidence.clamp(0.0, 1.0)

    raw_loss = -(labels * F.log_softmax(lambda_logits, dim=-1)).sum(dim=-1)
    zero = torch.zeros_like(raw_loss)
    unweighted_per_sample = torch.where(valid, raw_loss, zero)
    weighted_per_sample = torch.where(valid, raw_loss * confidence, zero)
    valid_confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
    valid_count = valid.sum().clamp_min(1)
    loss = weighted_per_sample.sum() / valid_confidence.sum().clamp_min(1e-6)
    unweighted_loss = unweighted_per_sample.sum() / valid_count
    confidence_mean = valid_confidence.sum() / valid_count
    stats = {
        "lambda_loss_unweighted": unweighted_loss.detach().item(),
        "lambda_confidence_mean": confidence_mean.detach().item(),
    }
    return loss, weighted_per_sample, stats
```

Update `compute_lambda_condition`:

```python
        pred = lambda_hat.detach()
        if lambda_t is None or lambda_is_valid is None:
            return pred

        labels = lambda_t.to(device=lambda_hat.device, dtype=lambda_hat.dtype)
        if labels.ndim == 1:
            labels = torch.stack([1.0 - labels, torch.zeros_like(labels), labels], dim=-1)
        labels = labels.view_as(lambda_hat)
        labels = labels.clamp_min(0.0)
        labels = labels / labels.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        valid = lambda_is_valid.to(device=lambda_hat.device, dtype=torch.bool).view(lambda_hat.shape[0], 1)
        alpha = max(0.0, min(1.0, float(alpha)))
        mixed = (1.0 - alpha) * labels + alpha * pred
        return torch.where(valid, mixed, pred).detach()
```

- [ ] **Step 4: Run loss and condition tests to verify GREEN**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py::test_compute_lambda_condition_mixes_distribution_labels_and_stopped_prediction tests/policies/smolvla/test_smolvla_lambda_conditioning.py::test_compute_lambda_supervision_loss_uses_soft_label_cross_entropy_and_confidence -q
```

Expected result: PASS.

- [ ] **Step 5: Write failing test for adapter gate based on `1 - p_low`**

Append:

```python
def test_lambda_gate_uses_non_low_frequency_probability():
    with patch(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel", FakeSmolVLMWithExpertModel
    ):
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

        model = VLAFlowMatching(make_config())

    low = torch.tensor([[0.9, 0.1, 0.0]])
    high = torch.tensor([[0.1, 0.2, 0.7]])

    low_gate = model.compute_lambda_gate(low, dtype=torch.float32)
    high_gate = model.compute_lambda_gate(high, dtype=torch.float32)

    torch.testing.assert_close(low_gate.flatten(), torch.tensor([0.1]))
    torch.testing.assert_close(high_gate.flatten(), torch.tensor([0.9]))
    assert low_gate.item() < high_gate.item()
```

- [ ] **Step 6: Run gate test to verify RED**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py::test_lambda_gate_uses_non_low_frequency_probability -q
```

Expected result: FAIL because `compute_lambda_gate` currently flattens scalar lambda values.

- [ ] **Step 7: Update adapter conditioning and gate**

Change `LambdaVelocityAdapter` to accept a `condition_dim`:

```python
class LambdaVelocityAdapter(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int, condition_dim: int = 3):
        super().__init__()
        self.condition_dim = condition_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_size + condition_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, action_suffix_out: Tensor, lambda_condition: Tensor) -> Tensor:
        lambda_condition = lambda_condition.to(device=action_suffix_out.device, dtype=action_suffix_out.dtype)
        lambda_condition = lambda_condition.view(action_suffix_out.shape[0], 1, self.condition_dim)
        lambda_condition = lambda_condition.expand(-1, action_suffix_out.shape[1], -1)
        return self.net(torch.cat([action_suffix_out, lambda_condition], dim=-1))
```

Instantiate it with `condition_dim=3`.

Update `compute_lambda_gate`:

```python
    def compute_lambda_gate(self, lambda_condition: Tensor, dtype: torch.dtype) -> Tensor:
        lambda_condition = lambda_condition.to(dtype=dtype)
        if lambda_condition.ndim == 1:
            non_low = lambda_condition.clamp(0.0, 1.0)
        else:
            lambda_condition = lambda_condition.view(lambda_condition.shape[0], 3)
            non_low = (lambda_condition[:, 1] + lambda_condition[:, 2]).clamp(0.0, 1.0)
        if self.config.lambda_adapter_gate == "lambda":
            gate = non_low
        elif self.config.lambda_adapter_gate == "sigmoid":
            gate = torch.sigmoid(
                self.config.lambda_adapter_sigmoid_slope
                * (non_low - self.config.lambda_adapter_sigmoid_center)
            )
        else:
            raise ValueError("lambda_adapter_gate must be 'lambda' or 'sigmoid'")
        return gate[:, None, None]
```

- [ ] **Step 8: Run gate test to verify GREEN**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py::test_lambda_gate_uses_non_low_frequency_probability -q
```

Expected result: PASS.

- [ ] **Step 9: Commit Task 3**

Run:

```bash
git add src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py
git commit -m "feat: train smolvla lambda frequency distributions"
```

---

### Task 4: Model Prediction Plumbing And Dynamic Urgency

**Files:**
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Modify: `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`

- [ ] **Step 1: Write failing tests for 3-logit head and dynamic execution urgency**

Append:

```python
def test_compute_lambda_hat_from_prefix_returns_probs_and_logits():
    with patch(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel", FakeSmolVLMWithExpertModel
    ):
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

        model = VLAFlowMatching(make_config())

    prefix_out = torch.randn(2, 5, 8)
    prefix_mask = torch.ones(2, 5, dtype=torch.bool)
    probs, logits = model.compute_lambda_hat_from_prefix(prefix_out, prefix_mask)

    assert probs.shape == (2, 3)
    assert logits.shape == (2, 3)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6)


def test_dynamic_n_action_steps_uses_non_low_probability():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = object.__new__(SmolVLAPolicy)
    policy.config = make_config()
    policy.config.dynamic_n_action_steps = True
    policy.config.lambda_ema_beta = 0.0
    policy._lambda_smooth = None

    assert policy._update_dynamic_n_action_steps(torch.tensor([[0.9, 0.1, 0.0]])) == 4
    assert policy._update_dynamic_n_action_steps(torch.tensor([[0.0, 0.0, 1.0]])) == 1
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py::test_compute_lambda_hat_from_prefix_returns_probs_and_logits tests/policies/smolvla/test_smolvla_lambda_conditioning.py::test_dynamic_n_action_steps_uses_non_low_probability -q
```

Expected result: FAIL because the prefix helper returns one scalar and dynamic execution averages all values.

- [ ] **Step 3: Update lambda head to output logits and probabilities**

In `VLAFlowMatching.__init__`, change the final lambda head layer to:

```python
            nn.Linear(self.vlm_with_expert.config.text_config.hidden_size, 3),
```

Update `compute_lambda_hat_from_prefix`:

```python
    def compute_lambda_hat_from_prefix(self, prefix_out: Tensor, prefix_pad_masks: Tensor) -> tuple[Tensor, Tensor]:
        prefix_out = prefix_out.to(dtype=torch.float32)
        mask = prefix_pad_masks.to(device=prefix_out.device, dtype=prefix_out.dtype).unsqueeze(-1)
        pooled = (prefix_out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.lambda_head(pooled)
        probs = torch.softmax(logits, dim=-1)
        return probs, logits
```

Update training forward:

```python
        lambda_logits = None
        if self._lambda_prediction_enabled():
            lambda_hat, lambda_logits = self.compute_lambda_hat_from_prefix(prefix_out, prefix_pad_masks)
            lambda_condition = self.compute_lambda_condition(...)
```

Return:

```python
        return {"losses": losses, "lambda_hat": lambda_hat, "lambda_logits": lambda_logits}
```

Update `SmolVLAPolicy.forward` to read `lambda_logits = model_output.get("lambda_logits")` and call:

```python
                lambda_logits=lambda_logits,
```

Update inference:

```python
            prefix_lambda_hat, _ = self.compute_lambda_hat_from_prefix(prefix_outputs[0], prefix_pad_masks)
```

- [ ] **Step 4: Update dynamic execution scalar extraction**

In `SmolVLAPolicy._update_dynamic_n_action_steps`, replace scalar mean extraction with:

```python
        lambda_tensor = lambda_hat.detach().float()
        if lambda_tensor.ndim >= 2 and lambda_tensor.shape[-1] == 3:
            urgency = lambda_tensor[..., 1] + lambda_tensor[..., 2]
        else:
            urgency = lambda_tensor
        lambda_now = float(urgency.mean().clamp(0.0, 1.0).item())
```

- [ ] **Step 5: Run model plumbing tests to verify GREEN**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py::test_compute_lambda_hat_from_prefix_returns_probs_and_logits tests/policies/smolvla/test_smolvla_lambda_conditioning.py::test_dynamic_n_action_steps_uses_non_low_probability -q
```

Expected result: PASS.

- [ ] **Step 6: Update stale lambda-conditioning tests**

Remove or replace `test_embed_suffix_adds_lambda_token_before_action_tokens`, because the current implementation conditions the adapter directly and no longer adds a lambda token to the suffix. Keep coverage through `test_lambda_gate_uses_non_low_frequency_probability` and `test_compute_lambda_condition_mixes_distribution_labels_and_stopped_prediction`.

Update `test_default_peft_targets_keep_lambda_modules_trainable` assertion:

```python
    assert "lambda_head" in targets["modules_to_save"]
    assert "lambda_velocity_adapter" in targets["modules_to_save"]
```

- [ ] **Step 7: Run full smolvla lambda-conditioning tests**

Run:

```bash
uv run pytest tests/policies/smolvla/test_smolvla_lambda_conditioning.py -q
```

Expected result: PASS.

- [ ] **Step 8: Commit Task 4**

Run:

```bash
git add src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py
git commit -m "feat: route smolvla frequency probabilities through model"
```

---

### Task 5: Final Verification And Cleanup

**Files:**
- Review: `src/lerobot/policies/smolvla/lambda_labels.py`
- Review: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Review: `src/lerobot/policies/smolvla/processor_smolvla.py`
- Review: `src/lerobot/policies/smolvla/configuration_smolvla.py`
- Review: `tests/policies/smolvla/test_lambda_labels.py`
- Review: `tests/policies/smolvla/test_smolvla_lambda_conditioning.py`
- Review: `tests/processor/test_smolvla_processor.py`

- [ ] **Step 1: Run targeted tests**

Run:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/processor/test_smolvla_processor.py -q
```

Expected result: PASS.

- [ ] **Step 2: Run targeted ruff**

Run:

```bash
uv run ruff check src/lerobot/policies/smolvla/lambda_labels.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/processor_smolvla.py tests/policies/smolvla/test_lambda_labels.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/processor/test_smolvla_processor.py
```

Expected result: PASS.

- [ ] **Step 3: Inspect diff for unintended unrelated changes**

Run:

```bash
git diff --stat HEAD
git diff -- src/lerobot/policies/smolvla/lambda_labels.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/processor_smolvla.py src/lerobot/policies/smolvla/configuration_smolvla.py tests/policies/smolvla/test_lambda_labels.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/processor/test_smolvla_processor.py
```

Expected result: only frequency-probability label, processor, model, config, and test changes are present. Existing unrelated user changes in `src/lerobot/configs/policies.py` and `src/lerobot/policies/smolvla/note.md` are not staged or reverted.

- [ ] **Step 4: Commit verification cleanup when cleanup was required**

When Step 1 or Step 2 required cleanup, run:

```bash
git add src/lerobot/policies/smolvla/lambda_labels.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/processor_smolvla.py src/lerobot/policies/smolvla/configuration_smolvla.py tests/policies/smolvla/test_lambda_labels.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/processor/test_smolvla_processor.py
git commit -m "test: verify smolvla frequency probability labels"
```

When both Step 1 and Step 2 pass without cleanup, do not create an empty commit.

---

## Self-Review

- Spec coverage: Task 1 covers DCT label generation and the requested low/mid/high band split. Task 2 covers sidecar and processor shape changes. Task 3 covers soft-label training loss and adapter gate using `1 - p_low`. Task 4 covers VLM prefix prediction and dynamic execution using `p_mid + p_high`. Task 5 covers targeted verification.
- Red-flag scan: This plan contains no unresolved markers, no open-ended implementation steps, and no test steps without explicit commands.
- Type consistency: `lambda_t` is `[N, 3]` in sidecars and `[B, 3]` in batches. `lambda_hat` is probabilities `[B, 3]`; `lambda_logits` is logits `[B, 3]`; `lambda_confidence` remains `[B]`; `lambda_is_valid` remains `[B]`.
