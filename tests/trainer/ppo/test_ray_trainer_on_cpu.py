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

import numpy as np

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _compute_self_distillation_source_flags
from verl.trainer.ppo.metric_utils import process_validation_metrics


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


def test_collect_reward_info_strings_and_extra_info_name_helpers():
    values = RayPPOTrainer._collect_reward_info_strings(
        reward_extra_infos_dict={"theme": ["lab_guidelines", None, "  rapport  "]},
        batch_size=4,
        key="theme",
    )

    assert values == ["lab_guidelines", "", "rapport", ""]
    assert RayPPOTrainer._theme_name_from_extra_info({"theme": " form_fill "}) == "form_fill"
    assert RayPPOTrainer._scenario_name_from_extra_info({"scenario": " question_answering "}) == "question_answering"
    assert RayPPOTrainer._matches_target_scenario(
        {"theme": " form_fill ", "scenario": " record_answer "},
        {"record_answer"},
    )
    assert RayPPOTrainer._matches_target_scenario(
        {"theme": " form_fill ", "scenario": " record_answer "},
        {"form_fill/record_answer"},
    )
    assert not RayPPOTrainer._matches_target_scenario(
        {"theme": " form_fill ", "scenario": " record_answer "},
        {"first_30s/identity_verification_failed"},
    )


def test_collect_reward_info_strings_supports_feedback_gate_and_validation_calls():
    reward_info = {
        "failed_verifier_trace_json": ["  trace-a  ", None, "[]", "trace-b"],
        "theme": ["form_fill", "", None, "  rapport  "],
    }

    # Validation-style positional call.
    theme_values = RayPPOTrainer._collect_reward_info_strings(reward_info, 4, "theme")
    assert theme_values == ["form_fill", "", "", "rapport"]

    # SRPO-style keyword call with environment feedback enabled.
    trace_values = RayPPOTrainer._collect_reward_info_strings(
        reward_extra_infos_dict=reward_info,
        batch_size=4,
        key="failed_verifier_trace_json",
        include_environment_feedback=True,
    )
    assert trace_values == ["trace-a", "", "", "trace-b"]

    # SRPO-style keyword call with environment feedback disabled.
    gated_values = RayPPOTrainer._collect_reward_info_strings(
        reward_extra_infos_dict=reward_info,
        batch_size=4,
        key="failed_verifier_trace_json",
        include_environment_feedback=False,
    )
    assert gated_values == ["", "", "", ""]


def test_validation_metric_breakdowns_include_theme_and_scenario_sections():
    sample_uids = ["uid1", "uid1", "uid2", "uid2"]
    infos_dict = {
        "reward": [0.9, 0.1, 0.3, 0.7],
        "scenario_score": [1.0, 0.0, 1.0, 1.0],
    }
    data_sources = np.array(["verifier_training_raw0417"] * 4, dtype=object)
    themes = ["lab_guidelines", "lab_guidelines", "rapport", "rapport"]
    scenarios = ["clarify_vague_value", "clarify_vague_value", "witty", "witty"]

    metric_dict = {}
    RayPPOTrainer._extend_validation_metric_section(
        metric_dict,
        section_prefix="val",
        grouped_metrics=process_validation_metrics(data_sources, sample_uids, infos_dict),
    )

    theme_labels, has_named_theme = RayPPOTrainer._build_validation_theme_labels(data_sources, themes)
    assert has_named_theme is True
    RayPPOTrainer._extend_validation_metric_section(
        metric_dict,
        section_prefix="val-theme",
        grouped_metrics=process_validation_metrics(theme_labels, sample_uids, infos_dict),
    )

    scenario_labels, has_named_scenario = RayPPOTrainer._build_validation_scenario_labels(
        data_sources, themes, scenarios
    )
    assert has_named_scenario is True
    RayPPOTrainer._extend_validation_metric_section(
        metric_dict,
        section_prefix="val-scenario",
        grouped_metrics=process_validation_metrics(scenario_labels, sample_uids, infos_dict),
    )

    assert metric_dict["val-core/verifier_training_raw0417/reward/mean@2"] == 0.5
    assert metric_dict["val-theme-core/verifier_training_raw0417/lab_guidelines/reward/mean@2"] == 0.5
    assert (
        metric_dict[
            "val-scenario-core/verifier_training_raw0417/lab_guidelines/clarify_vague_value/reward/mean@2"
        ]
        == 0.5
    )
    assert (
        metric_dict[
            "val-scenario-aux/verifier_training_raw0417/rapport/witty/scenario_score/mean@2"
        ]
        == 1.0
    )
