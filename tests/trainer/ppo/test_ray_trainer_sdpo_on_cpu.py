import types

import numpy as np
import pytest
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _make_trainer() -> RayPPOTrainer:
    return object.__new__(RayPPOTrainer)


def _make_batch(*uids: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(non_tensor_batch={"uid": np.asarray(uids, dtype=object)})


def test_collect_solutions_by_uid_respects_scenario_gate():
    trainer = _make_trainer()
    batch = _make_batch("u1", "u1", "u2")
    reward_tensor = torch.tensor([[0.97], [0.96], [0.94]], dtype=torch.float32)
    response_texts = ["candidate-a", "candidate-b", "candidate-c"]

    success_by_uid, metrics = trainer._collect_solutions_by_uid(
        batch=batch,
        reward_tensor=reward_tensor,
        response_texts=response_texts,
        success_reward_threshold=0.95,
        reward_extra_infos_dict={"scenario_score": [0.90, 1.00, 1.00]},
        solution_gate_min_scenario_score=1.0,
    )

    assert success_by_uid["u1"] == [1]
    assert success_by_uid["u2"] == []
    assert metrics["self_distillation/solution_candidate_fraction"] == pytest.approx(1 / 3)
    assert metrics["self_distillation/solution_rejected_by_scenario_fraction"] == pytest.approx(1 / 3)
    assert metrics["self_distillation/solution_rejected_by_reward_fraction"] == pytest.approx(1 / 3)


def test_collect_solutions_by_uid_respects_veto_and_length_gates():
    trainer = _make_trainer()
    batch = _make_batch("u1", "u2", "u3")
    reward_tensor = torch.tensor([[0.98], [0.97], [0.96]], dtype=torch.float32)
    response_texts = ["candidate-a", "candidate-b", "candidate-c"]

    success_by_uid, metrics = trainer._collect_solutions_by_uid(
        batch=batch,
        reward_tensor=reward_tensor,
        response_texts=response_texts,
        success_reward_threshold=0.95,
        reward_extra_infos_dict={
            "has_veto_failure": [1.0, 0.0, 0.0],
            "response_length": [32, 0, 18],
        },
        solution_gate_require_no_veto_failure=True,
        solution_gate_min_response_length=1,
    )

    assert success_by_uid["u1"] == []
    assert success_by_uid["u2"] == []
    assert success_by_uid["u3"] == [2]
    assert metrics["self_distillation/solution_rejected_by_veto_fraction"] == pytest.approx(1 / 3)
    assert metrics["self_distillation/solution_rejected_by_response_length_fraction"] == pytest.approx(1 / 3)


def test_collect_solutions_by_uid_requires_reward_extra_info_for_scenario_gate():
    trainer = _make_trainer()
    batch = _make_batch("u1")
    reward_tensor = torch.tensor([[1.0]], dtype=torch.float32)
    response_texts = ["candidate-a"]

    with pytest.raises(ValueError, match="solution_gate_min_scenario_score"):
        trainer._collect_solutions_by_uid(
            batch=batch,
            reward_tensor=reward_tensor,
            response_texts=response_texts,
            success_reward_threshold=0.95,
            reward_extra_infos_dict={},
            solution_gate_min_scenario_score=1.0,
        )
