from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import compute_dynamic_n_action_steps
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
    with patch(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel", FakeSmolVLMWithExpertModel
    ):
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
    with patch(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel", FakeSmolVLMWithExpertModel
    ):
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


def test_lambda_alpha_schedule_reaches_end_value():
    with patch(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel", FakeSmolVLMWithExpertModel
    ):
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


def test_lambda_conditioning_defaults_off_and_auto_enables_for_label_or_dynamic_paths():
    base_cfg = make_config()
    base_cfg.lambda_conditioning = False
    base_cfg.__post_init__()
    assert base_cfg.lambda_conditioning is False

    label_cfg = make_config()
    label_cfg.lambda_conditioning = False
    label_cfg.lambda_labels_path = "lambda_labels.pt"
    label_cfg.__post_init__()
    assert label_cfg.lambda_conditioning is True

    dynamic_cfg = make_config()
    dynamic_cfg.lambda_conditioning = False
    dynamic_cfg.dynamic_n_action_steps = True
    dynamic_cfg.__post_init__()
    assert dynamic_cfg.lambda_conditioning is True


def test_default_peft_targets_keep_lambda_modules_trainable():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = object.__new__(SmolVLAPolicy)
    targets = policy._get_default_peft_targets()

    assert "lambda_head" in targets["modules_to_save"]
    assert "lambda_token_mlp" in targets["modules_to_save"]


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
