#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_smolvla import SmolVLAConfig
from .lambda_labels import LambdaLabelLookup, load_lambda_sidecar


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
            complementary_data["lambda_confidence"] = torch.tensor(0.0, dtype=torch.float32)
            complementary_data["lambda_is_valid"] = torch.tensor(False, dtype=torch.bool)
            return complementary_data

        index = torch.as_tensor(complementary_data["index"], dtype=torch.long)
        lambda_t, lambda_confidence, lambda_is_valid = self._lookup.lookup_with_confidence(index)
        complementary_data["lambda_t"] = lambda_t
        complementary_data["lambda_confidence"] = lambda_confidence
        complementary_data["lambda_is_valid"] = lambda_is_valid
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {"labels_path": self.labels_path, "default_value": self.default_value}

    def transform_features(self, features):
        return features


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        NewLineTaskProcessorStep(),
    ]
    if config.lambda_labels_path is not None:
        input_steps.append(
            SmolVLALambdaLabelProcessorStep(
                labels_path=config.lambda_labels_path,
                default_value=config.lambda_default_value,
            )
        )
    input_steps.extend(
        [
            TokenizerProcessorStep(
                tokenizer_name=config.vlm_model_name,
                padding=config.pad_language_to,
                padding_side="right",
                max_length=config.tokenizer_max_length,
            ),
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
        ]
    )
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
