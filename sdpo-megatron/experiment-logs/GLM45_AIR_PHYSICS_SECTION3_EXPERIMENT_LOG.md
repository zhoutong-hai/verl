# GLM-4.5-Air Physics Section 3 Experiment Log

## Goal

Extend the current Physics Section 3 SDPO and GRPO experiment setup to `GLM-4.5-Air`, while preserving the same `verl`-native benchmark structure used for the Qwen Physics runs:

- SciKnowEval `physics`
- Megatron training path
- validation sample printing and dumps
- W&B tracking
- comparable SDPO and GRPO launcher variants

## Working instruction

- Continue working this experiment end-to-end until the model training is in a good state.
- Do not stop at setup, launch, or the first failure.
- If the run fails or looks unhealthy, diagnose it, make the needed fix, relaunch, and keep iterating until the failure is resolved.
- Do not stop after reporting a failure or a likely cause; carry the work through the fix and the next relaunch.
- Once the run appears healthy, keep monitoring it in case another failure or instability appears later.
- Do not pause and wait for another user instruction during that process unless there is a genuinely high-risk or irreversible decision.
- Treat this log as the running record for the full bring-up, not just a launch note.

## Why this model is next

- It is the next model of interest after the Qwen Physics bring-up.
- It likely requires a real multi-node Megatron setup rather than the smaller-model single-node path.
- The shared model path already exists on cluster storage:
  - `/hai/shared/pretrained_models/GLM-4.5-Air`

## Initial plan

1. Reuse the current Physics benchmark family instead of switching to a different dataset or training recipe.
2. Reuse the current `verl` Megatron experiment structure rather than the ms-swift setup.
3. Add a GLM-Air-specific Physics launcher that supports:
   - `VARIANT=sdpo_megatron`
   - `VARIANT=grpo_megatron`
4. Start with the stable `teacher_topk` SDPO path for bring-up.
5. Keep the current logging pattern:
   - inline validation samples
   - capped validation JSONL dumps
   - W&B tracking

## Relevant references

- Shared model path:
  - `/hai/shared/pretrained_models/GLM-4.5-Air`
- Current working Physics Megatron reference:
  - [QWEN3_8B_PHYSICS_SECTION3_EXPERIMENT_LOG.md](/Users/zhoutong/code/verl/sdpo-megatron/experiment-logs/QWEN3_8B_PHYSICS_SECTION3_EXPERIMENT_LOG.md)
- Closest `verl` GLM-Air model reference:
  - [test_dapo_glm_air_megatron.sh](/Users/zhoutong/code/verl/recipe/dapo/test_dapo_glm_air_megatron.sh)
- New GLM-Air SkyPilot task:
  - [verl-glm45-air-section3-physics.yaml](/Users/zhoutong/code/verl/examples/skypilot/verl-glm45-air-section3-physics.yaml)

## Bring-up findings before launch

### [Resolved] The shared model path is real and includes tokenizer/chat-template assets

Verified on an existing cluster node:

- `/hai/shared/pretrained_models/GLM-4.5-Air/config.json`
- `/hai/shared/pretrained_models/GLM-4.5-Air/tokenizer.json`
- `/hai/shared/pretrained_models/GLM-4.5-Air/tokenizer_config.json`
- `/hai/shared/pretrained_models/GLM-4.5-Air/chat_template.jinja`

The tokenizer loads successfully and already exposes a chat template.

### [Resolved] The current vLLM image appears to support `glm4_moe`

Inspected the installed vLLM package on the cluster node and found:

- `vllm/model_executor/models/glm4_moe.py`

So this should not hit the OLMo-style unsupported-model fallback issue.

### [Resolved] The current Megatron-Bridge path recognizes GLM-4.5-Air

Verified on the cluster node:

- `AutoBridge.from_hf_pretrained('/hai/shared/pretrained_models/GLM-4.5-Air', trust_remote_code=True)`
- `bridge.to_megatron_provider(load_weights=False)`

This succeeded and returned:

- `GLMMoEModelProvider`

That means the main worker path used by the current Megatron experiments should be viable for this model.

### [Noted] The local fallback registry is still incomplete for `Glm4MoeForCausalLM`

The repo-local `hf_to_mcore_config(...)` fallback path still fails with:

- `KeyError <SupportedModel.GLM4_MOE: 'Glm4MoeForCausalLM'>`

because the local Megatron registry includes forward-function entries for `GLM4_MOE` but not the corresponding config-converter / initializer registrations.

Current interpretation:

- this is not an immediate blocker for the planned bring-up path because the current Megatron experiments use Megatron-Bridge with `vanilla_mbridge=false`
- but it is still a cleanup item if we want the local non-Bridge fallback path to support GLM-Air cleanly

## New experiment artifacts

### Launcher file

- [verl-glm45-air-section3-physics.yaml](/Users/zhoutong/code/verl/examples/skypilot/verl-glm45-air-section3-physics.yaml)

### Config strategy

- The multi-node launcher now follows the single-file SkyPilot pattern.
- Model-specific GLM-Air overrides live directly in the SkyPilot `run:` block.
- The task launches `python3 -m verl.trainer.main_ppo` against the shared base configs:
  - `sdpo_megatron_trainer.yaml`
  - `ppo_megatron_trainer.yaml`
- The older wrapper shell script and GLM-Air-specific trainer YAMLs were removed as unused once the inline SkyPilot path became the canonical launcher.

## Current launch shape

- benchmark: SciKnowEval `physics`
- model: `GLM-4.5-Air`
- training path: Megatron

## Latest bring-up findings

### [Resolved Locally] GLM-Air MTP is incompatible with packed Megatron log-prob computation

Observed on reserved 4-node cluster `verl-glm45-air-physics`, job `21`.

Symptoms:

- The job got past Ray startup, official Megatron-Bridge export, and vLLM wake-up.
- It then failed during actor/ref `compute_log_prob()` with:
  - `AssertionError: multi token prediction + sequence packing is not yet supported.`
- Trace ended in:
  - `megatron/core/transformer/multi_token_prediction.py`
  - assertion on `packed_seq_params is None`

Diagnosis:

- GLM-4.5-Air exposes `num_nextn_predict_layers=1`, so Megatron MTP remains enabled in the model config.
- Our Megatron actor/ref log-prob path was still using packed `thd` format because the launcher set:
  - `actor_rollout_ref.model.use_remove_padding=true`
- That combination is currently unsupported by Megatron-Core for MTP models.

Fix:

- Disable remove-padding / sequence packing for this GLM-Air launcher:
  - `actor_rollout_ref.model.use_remove_padding=false`
- This should force the actor/ref path back to `bshd` instead of packed `thd`, avoiding the MTP assertion while preserving the GLM-Air model architecture and weights.

Why this fix:

- It is lower risk than trying to disable MTP in the model config or modify the checkpoint.
- GLM-Air bridge conversion already supports the MTP architecture, so keeping the model intact is safer for bring-up.
- expected cluster shape: `4 x H200:8`
- initial SDPO mode: `teacher_topk`
- GRPO baseline variant included in the same launcher path

Initial Megatron partition defaults in the new launcher:

- actor/ref TP = `2`
- actor/ref PP = `2`
- actor/ref CP = `2`
- actor/ref EP = `4`
- actor/ref ETP = `1`
- rollout TP = `8`

These are initial bring-up defaults derived from the closest GLM-Air Megatron reference in this repo and should be treated as tunable if the first run exposes memory or throughput issues.

## Commands

### SDPO

```bash
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
export WANDB_API_KEY='***'
sky launch -c verl-glm45-air-physics \
  /Users/zhoutong/code/verl/examples/skypilot/verl-glm45-air-section3-physics.yaml \
  --env VARIANT=sdpo_megatron \
  --secret WANDB_API_KEY -y
```

### GRPO

```bash
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
export WANDB_API_KEY='***'
sky launch -c verl-glm45-air-physics \
  /Users/zhoutong/code/verl/examples/skypilot/verl-glm45-air-section3-physics.yaml \
  --env VARIANT=grpo_megatron \
  --secret WANDB_API_KEY -y
```

## Monitoring metrics

- `val-core/sciknoweval/acc/mean@16`
- `val-core/sciknoweval/acc/best@16/mean`
- `val-core/sciknoweval/acc/maj@16/mean`
- `self_distillation/success_group_fraction`
- `self_distillation/reprompt_sample_fraction`
- `self_distillation/empty_target_batch`
- `critic/score/mean`
- `response_length/mean`
- `response_length/clip_ratio`
- inline validation samples in `run.log`
- capped validation dump files under `/hai/zhoutong/section3_physics_assets/validation_generations/`

## Current status

- The GLM-Air Physics experiment scaffolding is now prepared in `verl`.
- The main pre-launch risks have been checked:
  - model path exists
  - tokenizer/chat template exists
  - Megatron-Bridge recognizes the model
  - vLLM appears to support `glm4_moe`
- The next step is the first real 4-node bring-up launch.

## Bring-up progress

### [Resolved Locally] Hydra override mismatch in the single-YAML SDPO launcher

The first multi-node relaunch failed before model init because the launcher passed:

- `actor_rollout_ref.actor.self_distillation.teacher_scoring_mode=trainer_ref`

against `sdpo_megatron_trainer.yaml`, where this field is not declared in the base YAML struct even though it exists in the dataclass schema.

Fix:

- switch that one override to:
  - `+actor_rollout_ref.actor.self_distillation.teacher_scoring_mode=trainer_ref`

This allowed the run to pass Hydra config validation and enter real worker bring-up.

### [Resolved Locally] GLM-Air MoE dispatcher mismatch for non-DeepEP bring-up

The next bring-up failure happened during Megatron model construction:

- `AssertionError: DeepEP is not enabled. Please set --moe-enable-deepep to use DeepEP backend.`

Cause:

- the copied Qwen-style override set:
  - `moe_token_dispatcher_type=flex`
  - `moe_enable_deepep=false`
- on this stack, `flex` requires DeepEP
- but the intended bring-up path is the simpler non-DeepEP route

Fix:

- change the GLM-Air launcher to:
  - `moe_token_dispatcher_type=alltoall`
  - keep `moe_enable_deepep=false`

Current interpretation:

- this is a launcher/model-config issue, not a generic GLM-Air incompatibility
- the next relaunch should test whether the model can initialize cleanly on the non-DeepEP path

### [Resolved Locally] vLLM rollout GPU-memory target was too aggressive for colocated GLM-Air bring-up

After the MoE dispatcher fix, the run got through Megatron model construction but failed during vLLM rollout initialization with:

- `ValueError: Free memory on device (...) is less than desired GPU memory utilization (...)`

Observed on startup:

- free memory per rollout device was about `43.6 GiB`
- requested utilization at `VLLM_GPU_MEM_UTIL=0.35` implied about `48.9 GiB`

Cause:

- the copied rollout-memory target was too aggressive for the colocated 4-node GLM-Air setup after actor/model init had already reserved memory

Fix:

- lower `VLLM_GPU_MEM_UTIL` in the single SkyPilot launcher from `0.35` to `0.28`

Current interpretation:

- this is a rollout-capacity tuning issue, not a fundamental model-init failure
- the next relaunch should test whether vLLM can start cleanly with the lower memory target

### [In Progress] Widening rollout-memory headroom and clearing stale queued jobs

The next relaunch did not immediately start training. Instead, the cluster stayed `UP` while the new job remained stuck at:

- `Waiting for task resources on 4 nodes.`

At the same time, the prior failed job still showed the old vLLM startup failure at `VLLM_GPU_MEM_UTIL=0.35`, and Sky's managed-job controller looked flaky enough that queue inspection was unreliable.

Current action:

- widen the rollout-memory margin again by lowering `VLLM_GPU_MEM_UTIL` from `0.28` to `0.25`
- cancel stale queued jobs on the already-reserved cluster
- relaunch cleanly on the same 4-node reservation

Current interpretation:

- the underlying GLM-Air bring-up blockers have become narrower
- the next useful test is a clean relaunch with more rollout headroom, not another configuration redesign

### [In Progress] GLM-Air now reaches full actor/ref weight load before rollout startup

The next relaunch advanced significantly further:

- setup completed on all 4 nodes
- Ray head and workers joined
- W&B login succeeded
- `main_ppo` and the Megatron worker stack started
- actor/ref partitions began loading weights successfully

The new failure point remained in vLLM rollout startup, but much later than before:

- `ValueError: Free memory on device (34.51-34.54 / 139.8 GiB) on startup is less than desired GPU memory utilization (0.25, 34.95 GiB).`

Interpretation:

- the previous fixes are holding
- GLM-Air now gets through model construction and weight loading
- the remaining blocker is a narrow rollout-memory budget miss, not a structural model-init incompatibility

Current action:

- lower `VLLM_GPU_MEM_UTIL` again from `0.25` to `0.22`
- reduce the copied rollout/training `max_model_len` buffer from `18944` to `12288`, which is still above the current `2048 + 8192` prompt/response budget
- relaunch on the same reserved 4-node cluster

Update after job 9:

- `VLLM_GPU_MEM_UTIL=0.22` got past the earlier "free memory is less than desired utilization" gate
- but rollout still failed later during vLLM KV-cache initialization with `ValueError: No available memory for the cache blocks`
- this implies `0.22` is too low in the opposite direction: the admitted rollout budget is now below what GLM-Air needs to leave any KV-cache blocks after weights are resident

Next action:

- raise `VLLM_GPU_MEM_UTIL` to `0.24`
- tighten `max_model_len` from `12288` to `10240`, matching the current `2048 + 8192` prompt/response budget and the existing `max_reprompt_len=10240`
- relaunch immediately on the same reserved 4-node cluster

Additional launcher hardening:

- make `WANDB_API_KEY` optional in the SkyPilot YAML (`""` by default) so relaunches do not block on a missing local secret
- when a key is present, keep the existing `['console', 'wandb']` logger path
- when a key is absent, fall back to `['console']` so multi-node bring-up can continue and we can still debug from node-local logs

Update after job 11:

- the new rollout args were correctly reaching vLLM for `gpu_memory_utilization=0.24` and `max_num_batched_tokens=10240`
- but rollout still failed because `vllm_async_server.py` was overwriting `config.max_model_len` with `hf_config.max_position_embeddings`
- for GLM-Air that forced `--max_model_len 131072`, so vLLM sized KV cache against the full model context instead of the intended rollout cap
- observed failure:
  - `To serve at least one request with the model's max seq len (131072) ... available KV cache memory (0.88 GiB)`

Fix:

- respect an explicit rollout `max_model_len`
- only fall back to `hf_config.max_position_embeddings` when the rollout config leaves `max_model_len` unset
- still clamp explicit values so they never exceed the model's true positional limit

### [Resolved Locally] Disable NCCL NVLS for the 4-node GLM-Air Megatron bring-up

After the explicit `max_model_len` fix landed, the next relaunch got through:

- 4-node Ray startup
- Megatron actor/ref initialization
- GLM-Air weight loading
- vLLM server launch with the intended rollout limits

Verified effective vLLM serve args:

- `--max_model_len 10240`
- `--max_num_batched_tokens 10240`
- `--gpu_memory_utilization 0.24`

The next blocker appeared later in distributed startup as repeated NCCL warnings:

- `transport/nvls.cc:598 NCCL WARN Cuda failure 1 'invalid argument'`

Why this is likely the right fix:

- the closest known-good 4-node Megatron launcher in this repo already disables NVLS explicitly:
  - [verl-sdpo-megatron-llama33-4nodes.yaml](/Users/zhoutong/code/verl/examples/skypilot/verl-sdpo-megatron-llama33-4nodes.yaml)
- the GLM-Air launcher had not yet set `NCCL_NVLS_ENABLE`
- the failure surfaced only after the earlier rollout-memory and `max_model_len` issues were already fixed

Fix applied to the GLM-Air launcher:

- add `NCCL_NVLS_ENABLE: "0"` to `envs:`
- add `export NCCL_NVLS_ENABLE=0` in the `run:` block

Next step:

- relaunch on the same reserved 4-node cluster
- keep monitoring until the run either reaches the first real rollout/training step or exposes the next concrete blocker

### [Resolved Locally] Allow replicated PP-rank tensor specs during GLM-Air Megatron-Bridge export

After disabling NVLS, the next relaunch advanced further again:

- 4-node Ray startup succeeded
- Megatron actor/ref workers initialized
- GLM-Air actor/ref weights loaded
- vLLM servers started with the intended rollout limits
- training reached the first rollout wake-up before failing

The new blocker appeared in Megatron-Bridge during actor-to-vLLM weight export:

- `ValueError: Tensor exists on more than one PP rank. Found on ranks 0 and 1.`

Why this is likely the right interpretation:

- the failure comes from `megatron.bridge.models.conversion.param_mapping.broadcast_from_pp_rank`
- that helper assumes a converted tensor should exist on exactly one pipeline-parallel rank
- for GLM-Air, some exported tensors appear to be replicated across PP stages with the same tensor metadata
- the helper was crashing before it even checked whether the duplicated owners were actually compatible

Fix applied in the launcher setup patch:

- keep using the official `megatron-bridge` path (`vanilla_mbridge=false`)
- patch the installed `param_mapping.py` during `setup:`
- accept duplicated PP owners when their tensor specs are identical
- choose the first owner as the broadcast source
- log a warning with `cache_key` so the duplicated weight can still be identified later
- still fail loudly if multiple PP ranks report mismatched specs for the same tensor

Why not switch to `vanilla_mbridge=true` instead:

- the older `mbridge` package on this container is not compatible with the installed `megatron-core==0.15.0`
- direct import currently fails with:
  - `ModuleNotFoundError: No module named 'megatron.core.distributed.custom_fsdp'`

Next step:

- relaunch on the same reserved 4-node cluster with the patched official bridge
- keep monitoring until the run either reaches the first training step or exposes the next concrete bridge/cache-key issue

### [Resolved Locally] The GLM-Air MTP packing fix needed the Megatron engine flag, not just the model flag

The first attempt to work around GLM-Air's multi-token-prediction (MTP) incompatibility with sequence packing
disabled only:

- `actor_rollout_ref.model.use_remove_padding=false`

That was not sufficient. The next relaunch still failed in the Megatron actor/ref log-prob path with the same
assertion:

- `AssertionError: multi token prediction + sequence packing is not yet supported.`

Why the first fix was incomplete:

- the actual `thd` vs `bshd` decision in `megatron_actor.py` is controlled by
  `self.config.megatron.use_remove_padding`
- the actor config inherits that from `actor_rollout_ref.actor.megatron.use_remove_padding`
- the ref config inherits from the actor-side Megatron engine flag as well
- so the top-level model flag did not change the engine-level packing path used by `compute_log_prob()`

Correct fix:

- keep `actor_rollout_ref.model.use_remove_padding=false`
- also set:
  - `actor_rollout_ref.actor.megatron.use_remove_padding=false`
  - `actor_rollout_ref.ref.megatron.use_remove_padding=false`

Interpretation:

- GLM-Air can still use the same model weights and Megatron setup
- but actor/ref log-prob computation must run in non-packed `bshd` mode while MTP is enabled
- the previous failure was a config-plumbing mistake, not evidence that the broader GLM-Air path is blocked

Next step:

- relaunch again on the same reserved 4-node cluster with the engine-level no-packing override
- keep monitoring until the job either reaches real training steps or exposes the next concrete blocker

### [Resolved Locally] Non-packed GLM-Air actor/ref log-prob requires context parallel size 1

After the engine-level no-packing override landed, the next relaunch no longer hit:

- `AssertionError: multi token prediction + sequence packing is not yet supported.`

That confirmed the earlier MTP diagnosis was correct. The run then progressed into the non-packed Megatron
log-prob path and exposed the next assertion:

- `AssertionError: Context parallel size without seq_pack is not supported`

Where it failed:

- `verl/models/mcore/util.py::preprocess_bshd`
- the actor/ref log-prob path had switched from packed `thd` to non-packed `bshd`
- the launcher was still using `context_parallel_size=2`

Interpretation:

- this is not a new model incompatibility
- it is a direct consequence of the previous MTP fix:
  - GLM-Air with MTP cannot use packed actor/ref log-prob
  - current `verl` Megatron `bshd` preprocessing only supports `context_parallel_size=1`

Correct fix:

- keep the no-packing actor/ref path
- reduce actor/ref training `context_parallel_size` from `2` to `1`

Why this is the right next move:

- it aligns with the actual support matrix in the current `verl` Megatron path
- it is much smaller-risk than trying to add `bshd + CP` support during bring-up
- it preserves the main 4-node Megatron training shape while removing only the unsupported CP leg

Next step:

- relaunch immediately on the same reserved 4-node cluster with `TRAIN_CP_SIZE=1`
- continue monitoring until the run reaches real training steps or exposes the next concrete blocker

### [Resolved Locally] GLM-Air MTP must be disabled inside actor/ref RL forwards

After the `CP=1` fix, the run got past distributed bring-up and entered the real actor/ref `compute_log_prob()`
path. It then failed inside Megatron-Core's GLM multi-token-prediction branch with shape mismatches like:

- `RuntimeError: Sizes of tensors must match except in dimension 2. Expected size 5120 but got size 156`
- stack ending in:
  - `megatron/core/transformer/multi_token_prediction.py::_concat_embeddings`

Diagnosis:

- The earlier no-packing change removed the explicit packed-sequence assertion, but `GPTModel.forward()` still
  calls `_postprocess(..., mtp_in_postprocess=self.mtp_process)`.
- That means GLM-Air's MTP branch is still active during actor/ref `compute_log_prob()` and PPO forward passes,
  even though the RL objective only needs standard next-token logits.
- Current `verl` Megatron PPO/SDPO tensor plumbing is not compatible with GLM-Air's MTP auxiliary forward path.

Correct fix:

- Temporarily disable `mtp_process` on the unwrapped Megatron model during actor/ref RL forward passes, then
  restore it immediately afterward.
- This keeps the GLM-Air checkpoint and architecture intact while telling the RL path to use only the base
  next-token logits needed for PPO/SDPO.

Why this fix:

- It is lower risk than trying to retrofit full MTP support into the actor/ref RL path during bring-up.
- It avoids modifying the checkpoint or forcing a model-config mismatch at load time.
- For PPO/SDPO, ignoring the MTP auxiliary branch is acceptable because the RL loss is defined on the standard
  next-token distribution.

Next step:

- relaunch on the same reserved 4-node cluster with the actor/ref MTP-disable patch
- keep monitoring until the run reaches real training steps or exposes the next concrete blocker
