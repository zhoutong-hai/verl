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

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "NCCL_NVLS_ENABLE": "0",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "VLLM_USE_V1": "1",
        # symmetric memory allreduce not work properly in spmd mode
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "VLLM_USE_NCCL_SYMM_MEM": "0",
        "VLLM_ENABLE_PREFIX_CACHING": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES": "0",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}

# Training recipes often resolve config values from shell environment variables
# inside remote Ray actors, so keep the common launcher-side overrides in sync.
PPO_RAY_PASSTHROUGH_ENV_VARS = (
    "ACTOR_LR",
    "CUDA_VISIBLE_DEVICES",
    "FAILED_ATTEMPT_MAX_CHARS",
    "FORMAT_PENALTY",
    "HF_HOME",
    "INCLUDE_FAILED_ATTEMPT_IN_FEEDBACK_ONLY",
    "LCB_DATA_DIR",
    "MAX_MODEL_LEN",
    "MAX_PROMPT_LENGTH",
    "MAX_REPROMPT_LEN",
    "MAX_RESPONSE_LENGTH",
    "MODEL_PATH",
    "N_GPUS_PER_NODE",
    "NNODES",
    "PPO_MINI_BATCH_SIZE",
    "ROLLOUT_INCLUDE_STOP_STR",
    "ROLLOUT_PP_SIZE",
    "ROLLOUT_REPETITION_PENALTY",
    "ROLLOUT_STOP_STRINGS",
    "ROLLOUT_TP_SIZE",
    "SUCCESS_REWARD_THRESHOLD",
    "TRUNCATE_TO_FIRST_CODE_BLOCK",
    "VAL_INCLUDE_STOP_STR",
    "VAL_REPETITION_PENALTY",
    "VAL_STOP_STRINGS",
    "VALIDATION_DATA_DIR",
    "VALIDATION_ROOT",
    "VERL_REPO_DIR",
    "VLLM_GPU_MEM_UTIL",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
)


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    # Ray actors on remote nodes do not inherit the driver's shell environment.
    # Keep these variables in runtime_env, and prefer the driver's current value
    # when it exists so multi-node workers see the same transport/runtime setup.
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"][key] = os.environ[key]
    for key in PPO_RAY_PASSTHROUGH_ENV_VARS:
        if os.environ.get(key) is not None:
            runtime_env["env_vars"][key] = os.environ[key]
    return runtime_env
