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
import logging
import os
import re
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, prepare_raw_prompt, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

FIRST_CODE_BLOCK_PATTERN = re.compile(r"```(?:python|py)?\s*\n.*?```", re.DOTALL | re.IGNORECASE)


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.truncate_to_first_code_block = os.getenv("TRUNCATE_TO_FIRST_CODE_BLOCK", "0") == "1"

    def _maybe_truncate_to_first_code_block(self, output):
        if not self.truncate_to_first_code_block or not output.token_ids:
            return output.token_ids, output.log_probs

        response_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        match = FIRST_CODE_BLOCK_PATTERN.search(response_text)
        if match is None:
            return output.token_ids, output.log_probs

        truncated_text = match.group(0)
        if truncated_text.strip() == response_text.strip():
            return output.token_ids, output.log_probs

        truncated_token_ids = self.tokenizer.encode(truncated_text, add_special_tokens=False)
        if not truncated_token_ids:
            return output.token_ids, output.log_probs

        if output.log_probs is None:
            return truncated_token_ids, None

        # Preserve rollout logprobs only when the truncated code block can be expressed
        # as an exact prefix of the sampled token stream. If we cannot align the boundary
        # to the original tokens, keep the original sample rather than mixing rewritten
        # tokens with invalid logprobs.
        target_prefix_len = len(truncated_token_ids)
        candidate_prefix_lens = [target_prefix_len]
        for delta in range(1, 33):
            candidate_prefix_lens.extend([target_prefix_len - delta, target_prefix_len + delta])

        seen_prefix_lens = set()
        for prefix_len in candidate_prefix_lens:
            if prefix_len <= 0 or prefix_len > len(output.token_ids) or prefix_len in seen_prefix_lens:
                continue
            seen_prefix_lens.add(prefix_len)
            prefix_text = self.tokenizer.decode(output.token_ids[:prefix_len], skip_special_tokens=True)
            if prefix_text == truncated_text:
                return output.token_ids[:prefix_len], output.log_probs[:prefix_len]

        logger.debug("Normalizing to the first fenced code block by re-encoding and dropping rollout logprobs.")
        return truncated_token_ids, None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = prepare_raw_prompt(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        response_ids, response_logprobs = self._maybe_truncate_to_first_code_block(output)
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields={"rollout_metadata": output.metadata} if output.metadata else {},
        )
        return output
