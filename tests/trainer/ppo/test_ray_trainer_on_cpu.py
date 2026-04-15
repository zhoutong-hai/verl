# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _compute_self_distillation_source_flags


def test_compute_self_distillation_source_flags_default_uses_solution_or_feedback():
    flags = _compute_self_distillation_source_flags(
        self_distillation_eligible_list=[1.0, 1.0, 1.0, 0.0],
        solution_strs=["solution", None, None, "solution"],
        feedback_used=[False, True, False, True],
        failure_only_source_gate=False,
    )

    assert flags == [True, True, False, False]


def test_compute_self_distillation_source_flags_failure_only_requires_feedback():
    flags = _compute_self_distillation_source_flags(
        self_distillation_eligible_list=[1.0, 1.0, 1.0, 0.0],
        solution_strs=["solution", None, None, "solution"],
        feedback_used=[False, True, False, True],
        failure_only_source_gate=True,
    )

    assert flags == [False, True, False, False]


def test_collect_reward_info_scalars_tracks_missing_and_nonnumeric_values():
    values, available = RayPPOTrainer._collect_reward_info_scalars(
        reward_extra_infos_dict={"scenario_score": [1.0, "0.5", None, "bad"]},
        batch_size=5,
        key="scenario_score",
        default=0.0,
    )

    assert values == [1.0, 0.5, 0.0, 0.0, 0.0]
    assert available == [True, True, False, False, False]
