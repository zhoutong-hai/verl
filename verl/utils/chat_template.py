# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_QWEN3_TRAILING_AGENT_PREFIX_RE = re.compile(
    r"(?s)(<\|im_start\|>assistant\n)\s*Agent:\s*(?:<\|im_end\|>\s*)?\Z"
)


def _thinking_disabled(apply_chat_template_kwargs: dict) -> bool:
    """Return whether the caller explicitly requested Qwen non-thinking mode.

    Recent Qwen tokenizer versions use ``thinking=False`` while some earlier
    examples used ``enable_thinking=False``. We accept either knob so the raw
    string path stays aligned with the structured-message path.
    """
    return (
        apply_chat_template_kwargs.get("thinking") is False
        or apply_chat_template_kwargs.get("enable_thinking") is False
    )


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def normalize_string_prompt(prompt: str, **apply_chat_template_kwargs) -> str:
    """Normalize preformatted string prompts before tokenization.

    Some Qwen-formatted datasets serialize the final assistant cue as a closed turn
    like ``<|im_start|>assistant\\nAgent:<|im_end|>``. That boundary nudges Qwen3
    back into its thinking-format markers at generation time. When the caller
    explicitly disables thinking, normalize the trailing cue into the open
    generation-prefix form ``<|im_start|>assistant\\nAgent: ``.
    """
    if not isinstance(prompt, str):
        return prompt

    if _thinking_disabled(apply_chat_template_kwargs):
        return _QWEN3_TRAILING_AGENT_PREFIX_RE.sub(r"\1Agent: ", prompt)

    return prompt
