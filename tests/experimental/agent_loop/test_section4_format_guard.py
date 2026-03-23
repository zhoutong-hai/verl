import types

import numpy as np
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorkerBase, _InternalAgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        token_map = {
            1: "```python\n",
            2: "print('ok')",
            3: "\n```\nextra explanation",
        }
        return "".join(token_map[token_id] for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        if text == "```python\nprint('ok')\n```":
            return [10, 11, 12]
        raise AssertionError(f"Unexpected text for encode: {text!r}")


def _make_internal_output(response_logprobs):
    prompt_ids = torch.tensor([[11, 12]])
    response_ids = torch.tensor([[21, 22, 0, 0]])
    response_mask = torch.tensor([[1, 1, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    input_ids = torch.tensor([[11, 12, 21, 22, 0, 0]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    return _InternalAgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        position_ids=position_ids,
        response_mask=response_mask,
        attention_mask=attention_mask,
        response_logprobs=response_logprobs,
        routed_experts=None,
        multi_modal_inputs=None,
        multi_modal_data=None,
        reward_score=None,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={"reward_extra_info": {}},
    )


def test_single_turn_truncation_falls_back_when_logprobs_cannot_align():
    loop = object.__new__(SingleTurnAgentLoop)
    loop.tokenizer = FakeTokenizer()
    loop.truncate_to_first_code_block = True

    output = types.SimpleNamespace(token_ids=[1, 2, 3], log_probs=[-0.1, -0.2, -0.3])
    response_ids, response_logprobs = loop._maybe_truncate_to_first_code_block(output)

    assert response_ids == output.token_ids
    assert response_logprobs == output.log_probs


def test_postprocess_skips_rollout_log_probs_for_mixed_batches():
    worker = object.__new__(AgentLoopWorkerBase)
    outputs = [
        _make_internal_output(torch.tensor([[-0.1, -0.2, 0.0, 0.0]])),
        _make_internal_output(None),
    ]

    batch = AgentLoopWorkerBase._postprocess(worker, outputs)

    assert "rollout_log_probs" not in batch.batch.keys()
    assert np.array_equal(batch.non_tensor_batch["__num_turns__"], np.array([2, 2], dtype=np.int32))
