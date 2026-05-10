#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Read-side caches for LeRobotDataset."""

import datasets
import numpy as np
import torch


class DeltaColumnCache:
    """In-memory cache for numeric columns used by delta timestamp windows."""

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        delta_indices: dict[str, list[int]] | None,
        features: dict[str, dict],
        camera_keys: list[str],
    ) -> None:
        self._columns: dict[str, torch.Tensor] = {}
        if delta_indices is None:
            return

        for key in delta_indices:
            if key in camera_keys or key not in features or key not in hf_dataset.features:
                continue
            dtype = features[key].get("dtype")
            if dtype in {"image", "video", "string"}:
                continue
            try:
                column = hf_dataset.with_format("numpy", columns=[key])[:][key]
                if len(column) == 0:
                    continue
                if not isinstance(column, np.ndarray):
                    column = np.asarray(column)
                self._columns[key] = torch.as_tensor(column)
            except (TypeError, ValueError, RuntimeError, KeyError):
                continue

    def get(self, key: str, relative_indices: list[int]) -> torch.Tensor | None:
        tensor = self._columns.get(key)
        if tensor is None:
            return None
        index_tensor = torch.as_tensor(relative_indices, dtype=torch.long)
        return tensor.index_select(0, index_tensor)
