# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections.abc import Mapping
from typing import Any

_MEGATRON_LOSS_META_INFO_KEYS = (
    "debug_sdpo_dump_dir",
    "debug_sdpo_dump_step",
    "debug_sdpo_dump_once",
    "debug_sdpo_global_step",
    "debug_sdpo_experiment_name",
    "debug_sdpo_actor_strategy",
    "self_distillation_global_step",
)


def build_megatron_loss_meta_info(*, config: Mapping[str, Any], data_meta_info: Mapping[str, Any]) -> dict[str, Any]:
    """Build the compact meta_info payload forwarded into the Megatron loss closure."""

    meta_info = {
        "clip_ratio": config["clip_ratio"],
        "entropy_coeff": config["entropy_coeff"],
        "clip_ratio_c": config.get("clip_ratio_c", 3.0),
        "repair_ce_only": bool(data_meta_info.get("repair_ce_only", False)),
        "repair_ce_weight": float(data_meta_info.get("repair_ce_weight", 0.0) or 0.0),
    }
    for key in _MEGATRON_LOSS_META_INFO_KEYS:
        meta_info[key] = data_meta_info.get(key)
    return meta_info
