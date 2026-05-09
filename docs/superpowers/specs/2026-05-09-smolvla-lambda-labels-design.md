# SmolVLA Lambda Labels Design

## Goal

Add an offline lambda-label workflow for SmolVLA that trains a masked residual predictor, generates future-refinement labels in normalized action space, saves diagnostic tensors, and lets SmolVLA consume those labels during training while using predicted lambda values for dynamic closed-loop inference.

## Scope

This design implements the sidecar-label approach. The original LeRobotDataset parquet files and metadata are not modified. Offline tools live under `src/lerobot/policies/smolvla`, and SmolVLA training reads generated labels through configuration and preprocessing.

The first implementation targets 10-frame action chunks for datasets such as LIBERO where 1 second corresponds to 10 frames/actions. The anchor/query defaults are:

- `I_anchor = [0, 3, 6, 9]`
- `I_query = [1, 2, 4, 5, 7, 8]`
- `eps = 1e-6`

These values should be configurable, but validation must reject inconsistent values such as query indices overlapping anchors or indices outside the chunk.

## Offline Label Generation

Create `src/lerobot/policies/smolvla/lambda_labels.py` as the main implementation module. It should provide reusable functions/classes and a CLI-compatible entrypoint that can be wrapped from `src/lerobot/scripts` later if desired.

The module responsibilities are:

- Load a `LeRobotDataset` by `repo_id`, `root`, optional episodes, and revision.
- Extract sliding action chunks `A_t = [a_t, ..., a_{t+9}]` only for valid chunk starts inside each episode.
- Normalize actions using dataset action stats and the configured action normalization mode before any decomposition or error calculation.
- Build `A_trend` from fixed anchors with per-action-dimension PCHIP interpolation.
- Train a masked residual predictor in stage one.
- Freeze the residual predictor and generate labels in stage two.
- Save labels and diagnostics keyed by the dataset absolute `index` for the chunk start.

The saved sidecar file should be a `torch.save` dictionary with at least:

- `lambda_by_index`: mapping or tensor table from absolute dataset index to scalar lambda.
- `valid_indices`: absolute start indices included in the label file.
- `metadata`: repo id, root/revision if provided, action normalization mode, chunk size, anchors, query indices, model config, checkpoint path, creation timestamp.
- `metrics`: `e_trend`, `e_full`, `e_improve` per valid chunk.
- Optional diagnostics controlled by config: `A_t`, `A_trend`, `A_residual`, `A_hat`.

The default output should save scalar labels and metrics for all valid chunks. Diagnostic tensors are potentially large, so they should be optional or capped by a `max_diagnostic_chunks` setting.

## PCHIP Trend

PCHIP interpolation is applied independently per action dimension. The canonical implementation should use `scipy.interpolate.PchipInterpolator` when available. Because SciPy is behind extras in this repository, the tool must fail with a clear message instructing users to install a SciPy-enabled extra such as `lerobot[libero]`, `lerobot[pi]`, or `lerobot[scipy-dep]` if SciPy is missing.

The interpolation function should accept a normalized tensor or NumPy array shaped `[N, 10, D]` or `[10, D]` and return a tensor of the same shape. Anchor positions in `A_trend` should exactly match the original anchor actions up to numerical tolerance.

## Masked Residual Predictor

The residual predictor must not receive the full future action chunk. Its input is `A_trend`, anchors, or a compact representation derived from them. The simple default should be an MLP over flattened `A_trend`, producing query residuals shaped `[N, len(I_query), D]`.

During training:

- Target residual is `A_t[I_query] - A_trend[I_query]`.
- Loss is MSE over query positions and action dimensions.
- Anchor residual is never predicted as a learnable output.

During generation:

- Fill a full residual tensor with zeros.
- Scatter predicted query residuals into `I_query`.
- Keep `A_residual[I_anchor] = 0`.
- Compute `A_hat = A_trend + A_residual`.

This keeps anchors hard constrained and avoids future-action leakage into the residual branch.

## Lambda Definition

All errors are computed in normalized action space and only over query positions:

```text
e_trend = mean_{i in I_query,d} (A_t[i,d] - A_trend[i,d])^2
e_full = mean_{i in I_query,d} (A_t[i,d] - A_hat[i,d])^2
e_improve = e_trend - e_full
lambda_t = clamp(e_improve / (e_trend + 1e-6), 0, 1)
```

Each `lambda_t` belongs only to the chunk start sample `(o_t.image, o_t.state, text)`. The sidecar lookup must not assign the same label to interior chunk actions `a_t ... a_{t+9}` unless those interior frames are also valid chunk starts with their own independently computed labels.

## SmolVLA Training Integration

Add configuration fields to `SmolVLAConfig`:

- `lambda_labels_path: str | None = None`
- `lambda_loss_weight: float = 0.05`
- `lambda_loss_type: str = "smooth_l1"`
- `lambda_conditioning: bool = True`
- `lambda_alpha_start: float = 0.0`
- `lambda_alpha_end: float = 1.0`
- `lambda_alpha_warmup_steps: int = 30000`
- `lambda_default_value: float = 0.0`
- `dynamic_n_action_steps: bool = False`
- `dynamic_n_action_steps_min: int = 3`
- `dynamic_n_action_steps_max: int = 10`
- `lambda_ema_beta: float = 0.8`

The SmolVLA preprocessor should load `lambda_labels_path` once and use the batch `index` field to add:

- `lambda_t`: shape `[B]`, float32.
- `lambda_is_valid`: shape `[B]`, bool.

If a sample index has no label, `lambda_t` should use `lambda_default_value` and `lambda_is_valid=False`.

`VLAFlowMatching` should add:

- `lambda_head`: predicts `lambda_hat` from fused prefix hidden state.
- `lambda_token_mlp`: maps scalar `lambda_cond` to one expert hidden token.

Forward pass behavior:

- Compute the normal flow matching loss as before.
- Compute `lambda_hat` from prefix features.
- If valid labels exist, add `L_lambda` as SmoothL1 or MSE over valid samples.
- Use `lambda_cond=(1-alpha)*lambda_t + alpha*stopgrad(lambda_hat)` when labels are valid.
- Use `stopgrad(lambda_hat)` where labels are missing.
- Add the lambda token before action/timestep suffix tokens: `[lambda_token, noisy_action_tokens]`.
- Preserve compatibility when `lambda_conditioning=False` or no labels path is configured.

The existing `reduction="none"` path must remain compatible with sample weighting. Per-sample return should include flow loss plus weighted lambda auxiliary loss where valid labels exist.

The alpha schedule should be maintained inside the policy, likely through an `update()` hook that increments an internal train step after optimizer updates. It should also work when training resumes from checkpoints because the step buffer is part of the module state.

## Inference Integration

At inference there is no offline label. The model should use `lambda_hat` only.

When `dynamic_n_action_steps=False`, current behavior is unchanged.

When `dynamic_n_action_steps=True`:

```text
lambda_smooth = beta * lambda_prev + (1 - beta) * lambda_now
n_exec = round(n_max - lambda_smooth * (n_max - n_min))
```

`n_exec` is clamped to `[n_min, n_max]` and must not exceed `chunk_size`. In `select_action`, when a new chunk is sampled, enqueue only the first `n_exec` actions instead of the fixed `config.n_action_steps`. `predict_action_chunk` should continue returning the full chunk and may expose the latest predicted lambda in an attribute or optional metadata later; no API break is required for the initial implementation.

The EMA state resets on `policy.reset()`.

## Testing

Add focused tests under `tests/policies/smolvla`:

- PCHIP trend preserves anchors and returns expected smooth interpolation shape.
- Residual predictor output scatter keeps anchor residuals exactly zero.
- Lambda metrics use only query positions and clamp to `[0, 1]`.
- Sidecar lookup assigns labels only to exact chunk-start indices.
- SmolVLA lambda conditioning can be tested with a lightweight fake `VLAFlowMatching` or monkeypatched VLM dependency so tests do not download real SmolVLA weights.
- Dynamic execution step calculation clamps and EMA-smooths as expected.

The implementation should run targeted tests with `uv run pytest tests/policies/smolvla -q` where dependency availability allows. Unit tests that do not require transformers or CUDA should avoid skip decorators.

## Compatibility Notes

Existing SmolVLA configs should behave the same by default. All new training behavior is off unless `lambda_labels_path` or `lambda_conditioning` is configured. Existing RTC behavior remains separate; dynamic `n_action_steps` applies to `select_action`, while RTC users already use `predict_action_chunk`.

The sidecar label approach intentionally avoids changing LeRobotDataset schema, Hub dataset contents, or dataset stats. This makes labels reproducible and replaceable without invalidating the base dataset.
