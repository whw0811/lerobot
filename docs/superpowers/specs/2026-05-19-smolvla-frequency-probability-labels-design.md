# SmolVLA Frequency Probability Labels Design

## Goal

Replace SmolVLA scalar lambda supervision with a 3-way frequency probability label for each chunk start action `a_t`. The label is computed from the 10-step future action chunk `[a_t, ..., a_{t+9}]` using a cosine/DCT decomposition and represents how much future action energy lies in low, mid, and high temporal frequency bands.

The training model should predict the same 3-way distribution from VLM prefix hidden states. The adapter that refines `v_base` should be gated by non-low-frequency probability, so a high low-frequency probability constrains the adapter to make smaller corrections.

## Frequency Label Definition

For each valid 10-frame action chunk:

1. Normalize actions using the same action normalization path used by the existing lambda label workflow.
2. Compute a DCT-II-like cosine basis along the time dimension for every action dimension.
3. Convert coefficients to energy with squared coefficients and sum across action dimensions.
4. Aggregate frequency bands:
   - `low`: `k = 0, 1, 2`
   - `mid`: `k = 3, 4, 5`
   - `high`: `k = 6, 7, 8, 9`
5. Convert band energies to a probability vector:

```text
lambda_t = [p_low, p_mid, p_high]
p_band = energy_band / max(total_energy, eps)
```

If total energy is numerically zero, use the conservative default `[1, 0, 0]`, because a flat action chunk should not ask the adapter for strong correction.

The sidecar keeps the field name `lambda_t` for continuity, but its shape changes from scalar `[N]` to distribution `[N, 3]`.

## Sidecar And Processor

The sidecar remains a `torch.save` dictionary keyed by exact dataset `index` values. `valid_indices` remains 1D. `lambda_t` becomes float32 `[N, 3]`. `alpha_conf` can remain optional; if present, it remains a per-sample scalar confidence `[N]` and weights the auxiliary distribution loss.

`LambdaLabelLookup` should support both old scalar payloads and new distribution payloads:

- New payloads return `[B, 3]` probabilities.
- Old scalar payloads are converted to `[1 - lambda, 0, lambda]`.
- Missing indices return `lambda_default_distribution`, defaulting to `[1, 0, 0]`.
- `lambda_is_valid` remains `[B]`.

`SmolVLALambdaLabelProcessorStep` should inject:

- `lambda_t`: `[B, 3]`, float32.
- `lambda_confidence`: `[B]`, float32.
- `lambda_is_valid`: `[B]`, bool.

## Model Changes

`VLAFlowMatching.lambda_head` should predict 3 logits from pooled VLM prefix hidden states. The public prediction used by training stats and inference should be a probability distribution:

```text
lambda_hat = softmax(lambda_logits, dim=-1)
```

The auxiliary loss should use soft-label cross entropy over valid samples:

```text
L_lambda = -sum(lambda_t * log_softmax(lambda_logits))
```

When `lambda_confidence` is provided, weight valid per-sample losses by confidence and normalize by valid confidence sum.

`compute_lambda_condition` should mix label and stopped prediction distributions during warmup:

```text
lambda_cond = (1 - alpha) * lambda_t + alpha * stopgrad(lambda_hat)
```

For missing labels, use `stopgrad(lambda_hat)`.

## Adapter Gate

The velocity path remains:

```text
v_t = v_base + lambda_adapter_scale * gate(lambda_cond) * adapter(action_suffix_out, lambda_cond)
```

The gate is based on non-low-frequency probability:

```text
non_low = p_mid + p_high = 1 - p_low
```

With `lambda_adapter_gate="lambda"`, `gate = non_low`. With `lambda_adapter_gate="sigmoid"`, apply the existing sigmoid shaping to `non_low`. This makes low-frequency chunks constrain adapter correction instead of strengthening it.

`LambdaVelocityAdapter` should take the 3-way distribution as conditioning input rather than a scalar.

## Dynamic Execution

Existing dynamic action step selection can continue to use a scalar urgency derived from the predicted distribution:

```text
lambda_value = p_mid + p_high
```

This preserves the previous interpretation where larger values mean higher-frequency or less stable future motion and should shorten the execution horizon.

## Testing

Add focused tests before implementation:

- DCT frequency probability labels map pure low, mid, and high synthetic signals to the expected dominant band.
- Generated labels have shape `[N, 3]`, are non-negative, and each row sums to 1.
- Sidecar lookup returns `[B, 3]` labels and `[1, 0, 0]` for missing indices.
- Soft-label lambda loss accepts logits and 3-way labels, masks invalid samples, and applies confidence weighting.
- Adapter gate uses `1 - p_low`, so low-frequency labels produce smaller gates than high-frequency labels.
- Dynamic action step selection uses `p_mid + p_high` as the scalar urgency.

Targeted verification should include:

```bash
uv run pytest tests/policies/smolvla/test_lambda_labels.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/processor/test_smolvla_processor.py -q
uv run ruff check src/lerobot/policies/smolvla/lambda_labels.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/processor_smolvla.py tests/policies/smolvla/test_lambda_labels.py tests/policies/smolvla/test_smolvla_lambda_conditioning.py tests/processor/test_smolvla_processor.py
```

## Compatibility

Default SmolVLA behavior remains unchanged when lambda labels and lambda conditioning are disabled. Existing scalar lambda sidecars are read through the explicit migration mapping `[1 - lambda, 0, lambda]`. New generated sidecars always use the 3-way distribution.
