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

from verl.workers.actor_meta_info import build_megatron_loss_meta_info


def test_build_megatron_loss_meta_info_preserves_self_distillation_global_step():
    config = {
        "clip_ratio": 0.2,
        "entropy_coeff": 0.01,
        "clip_ratio_c": 2.5,
    }
    data_meta_info = {
        "self_distillation_global_step": 32,
        "repair_ce_only": True,
        "repair_ce_weight": 0.3,
        "debug_sdpo_dump_step": 7,
    }

    meta_info = build_megatron_loss_meta_info(config=config, data_meta_info=data_meta_info)

    assert meta_info["clip_ratio"] == 0.2
    assert meta_info["entropy_coeff"] == 0.01
    assert meta_info["clip_ratio_c"] == 2.5
    assert meta_info["repair_ce_only"] is True
    assert meta_info["repair_ce_weight"] == 0.3
    assert meta_info["debug_sdpo_dump_step"] == 7
    assert meta_info["self_distillation_global_step"] == 32
