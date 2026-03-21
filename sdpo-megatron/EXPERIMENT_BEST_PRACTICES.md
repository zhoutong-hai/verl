# SDPO Megatron Experiment Best Practices

Last updated: 2026-03-20

## Why this doc exists

This note is for future agents and operators who need to run, debug, and iterate on `verl` Megatron SDPO experiments efficiently.

It captures the working patterns that proved useful across:

- Qwen3 8B Physics small-model parity runs
- GLM-4.5-Air 4-node bring-up and follow-up runs
- SDPO, GRPO, `teacher_topk`, and `student_topk` comparisons

It is intentionally operational. The goal is to make the next experiment loop fast, reproducible, and hard to derail.

## Core principles

1. Prefer one canonical launcher per experiment family.
   - For multi-node jobs, keep the main launch flow in one SkyPilot YAML.
   - Avoid spreading the real launch contract across a YAML, a shell wrapper, and several hidden setup scripts unless there is a strong reason.

2. Separate bring-up from algorithm comparison.
   - First prove that the model, cluster, Ray, vLLM, and Megatron path run cleanly.
   - Only then compare `teacher_topk` vs `student_topk`, SDPO vs GRPO, KL ablations, or dataset swaps.

3. Use the cheapest stable reference first.
   - For a new large model, start from the most conservative path that already works on that stack.
   - In practice this usually means:
     - SDPO before GRPO when teacher guidance is available
     - `teacher_topk` before `student_topk`
     - shorter sequence budgets before larger ones

4. Treat early stability and late quality as different questions.
   - A run that passes step `1` or step `5` is only a bring-up success.
   - A run that stays healthy through the known collapse window is the real signal.

5. Do not stop at diagnosis.
   - If a run fails, keep going through:
     - diagnose
     - patch
     - relaunch
     - monitor
   - Only pause for user input when the next action is expensive, destructive, or changes experiment intent.

## Canonical experiment loop

1. Reserve the cluster early.
   - Large multi-node capacity is often the real bottleneck.
   - Reserve first, then iterate inside the reserved cluster.

2. Use SkyPilot as the canonical provisioning format.
   - Keep the checked-in single YAML as the source of truth.
   - Even if manual relaunches are used temporarily, the YAML should remain the authoritative config.

3. Reuse a healthy live cluster during bring-up.
   - For long-lived 4-node jobs, manual head-node relaunch is often much faster than waiting for full managed relaunch/setup.
   - This is an operational optimization, not the preferred final interface.

4. Keep experiment logs current while you work.
   - Record:
     - exact experiment name
     - W&B link
     - local log path
     - failure signature
     - fix
     - relaunch state
   - Do not rely on memory for long multi-failure loops.

5. Push code and doc changes once the fix is real.
   - Keep the branch aligned with the live cluster state.
   - Avoid hidden local-only hotfixes once a diagnosis is confirmed.

## Bring-up checkpoints

Use these checkpoints in order.

### Startup checkpoint

The run should get through:

- config validation
- dataset load/filter
- Ray worker bring-up
- Megatron actor/ref shard load
- vLLM startup
- W&B registration
- first `Training Progress` line

If it fails before this, treat it as an infra/integration bug, not an algorithm result.

### First-step checkpoint

The run should complete:

- `training/global_step = 1`

For SDPO, check:

- `self_distillation/success_group_fraction`
- `self_distillation/reprompt_sample_fraction`
- `self_distillation/empty_target_batch`
- `actor/grad_norm`
- `response_length/mean`
- `response_length/clip_ratio`

Healthy first-step signals usually look like:

- nonzero `grad_norm`
- active SDPO supervision
- `empty_target_batch` near `0`
- response length well below the cap

### First-validation checkpoint

The run should reach:

- step `5`
- first `val-core/sciknoweval/acc/mean@16`

This is the minimum point to call the run "working."

### Collapse-window checkpoint

The run should pass the known failure window for that experiment family.

Examples:

- Qwen Megatron SDPO used to fail before or around step `10-20`
- GLM-Air GRPO no-KL drifted badly by steps `33` and `55-59`
- GLM-Air KL-GRPO delayed collapse until around step `115`

Do not call a run healthy just because step `5` looks good if that family usually fails later.

## What to monitor

### Common training health

- `training/global_step`
- `actor/grad_norm`
- `response_length/mean`
- `response_length/clip_ratio`
- `training_ppl`
- `rollout_corr/kl`

### SDPO-specific

- `self_distillation/success_group_fraction`
- `self_distillation/reprompt_sample_fraction`
- `self_distillation/empty_target_batch`
- `self_distillation/per_token_loss_mean`
- `self_distillation/teacher_preferred_token_fraction`
- `self_distillation/selected_logprob_from_full_abs_diff_mean`

### Validation

Always compare the same family of metrics:

- `val-core/sciknoweval/acc/mean@16`
- `val-core/sciknoweval/acc/best@16/mean`
- `val-core/sciknoweval/acc/maj@16/mean`

Do not mix `mean@16`, `best@16`, and `maj@16` casually when comparing runs.

## Failure triage order

When a run fails, debug in this order.

### 1. Contract mismatches

Look for:

- shape mismatches
- packed-vs-response alignment errors
- `response_mask` vs packed-label disagreements
- missing batch fields

These are often faster to fix than they look and can completely invalidate metrics if left unresolved.

### 2. Probability-path correctness

For Megatron full-logit SDPO, confirm:

- selected-token log-prob path
- reconstructed full-logit path
- top-k support path

all describe the same distribution.

The Qwen collapse was ultimately caused by a real correctness bug here.

### 3. Multi-node environment propagation

On large jobs, confirm remote Ray actors inherit the same transport/runtime env as the head process.

Examples that mattered in practice:

- `NCCL_NVLS_ENABLE=0`
- `VLLM_ALLREDUCE_USE_SYMM_MEM=0`
- `VLLM_USE_NCCL_SYMM_MEM=0`
- `TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES=0`

### 4. Sequence-budget pressure

If the job dies or destabilizes later, check:

- prompt length distribution
- response length growth
- reprompt length
- actor/ref token budget
- rollout model length

The prompt is often not the real problem. Long responses and SDPO reprompts usually dominate memory.

### 5. Objective dynamics

Only after correctness and infra look sound should you conclude:

- the algorithm is unstable
- the reward is weak
- KL is needed
- `teacher_topk` vs `student_topk` is the key delta

## Sequence and memory budgeting

For large models, these are the first knobs to review.

### Highest memory impact

- `data.max_response_length`
- `actor_rollout_ref.rollout.max_model_len`
- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`
- `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`
- `actor_rollout_ref.actor.self_distillation.max_reprompt_len`

### Highest throughput impact

- `actor_rollout_ref.rollout.n`
- validation `n`
- `train_batch_size`
- `ppo_mini_batch_size`
- `ppo_micro_batch_size_per_gpu`
- rollout `gpu_memory_utilization`

### Parallelism/layout knobs

- tensor parallel size
- pipeline parallel size
- context parallel size
- expert parallel size
- rollout tensor parallel size

For new large-model bring-up:

- start conservative on sequence budgets
- prove stability
- only then expand context or response limits

## SDPO-specific guidance

### Default ordering for new model bring-up

1. SDPO `teacher_topk`
2. GRPO baseline
3. SDPO `student_topk`
4. KL / reward / dataset ablations

This keeps the first goal simple:

- prove the stack works with teacher-guided SDPO

before adding more semantic comparison axes.

### `teacher_topk` vs `student_topk`

Use `teacher_topk` when:

- you are bringing up a new model stack
- you want the simplest stable path
- you want to minimize moving parts

Use `student_topk` when:

- the base path is already stable
- you want closer parity with the current branch-local FSDP reproduction
- you are testing support-construction semantics rather than raw infrastructure

### Invariant to keep

For Megatron full-logit SDPO, keep watching:

- `self_distillation/selected_logprob_from_full_abs_diff_mean`

If this stops being tiny, treat it as a correctness regression first.

## GRPO-specific guidance

GRPO can look healthy early and still fail later through response-length inflation.

Observed pattern:

- early validation looks fine
- response length drifts upward
- clip ratio rises
- the run converges to the max-response-length regime
- quality degrades even when formatting stays correct

Practical implications:

- compare GRPO against SDPO on the same budget
- consider KL anchoring early if the model family is prone to long-answer drift
- watch collapse windows, not only step `5`

## Logging and traceability

Every experiment entry should record:

- experiment name
- date
- launcher/config variant
- W&B link
- local log path
- checkpoint/validation dump path if relevant
- first good checkpoint
- exact failure signature if any
- exact fix commit if any

If a fix is hypothesis-driven, say that clearly.
If a fix is confirmed by metrics or invariants, say that clearly too.

## Preferred operational patterns

### Use manual relaunch only as a speed tool

Manual relaunch on the head node is acceptable when:

- the cluster is already reserved
- Ray is healthy
- multi-node setup is expensive to repeat
- you need fast diagnose-fix-relaunch loops

But the checked-in SkyPilot YAML should still be the canonical config.

### Keep launchers clean

- pass `WANDB_API_KEY` via secret/env, not hardcoded files
- avoid launcher-only experiment-name branches unless needed
- keep run naming predictable
- remove obsolete wrapper scripts once a single-YAML path exists

### Avoid hidden setup drift

If setup requires compatibility patches:

- best: upgrade or pin the dependency version that already works
- good: apply a tracked repo-local patch script
- acceptable only for bring-up: inline site-package edits in the YAML

Do not leave critical compatibility fixes undocumented.

## Recommended experiment order for a new large model

1. Reserve cluster.
2. Launch conservative SDPO `teacher_topk`.
3. Get past:
   - startup
   - step `1`
   - step `5`
   - known collapse window
4. Launch GRPO baseline on the same cluster shape and sequence budget.
5. Launch SDPO `student_topk`.
6. Only then move to:
   - KL ablations
   - custom dataset
   - reward changes
   - larger context budgets

## What "good state" means

For this project family, a run is in a good state when:

- it is past the known startup failure boundaries
- it has passed the known collapse window for that setup
- response length is not saturating the cap
- clip ratio is not trending toward `1.0`
- gradients remain healthy
- validation is in the expected regime for that model/variant

A run that merely starts is not enough.
A run that survives the known failure regime and stays interpretable is the real target.

## Related docs

- [IMPLEMENTATION_DESIGN.md](/Users/zhoutong/code/verl/sdpo-megatron/IMPLEMENTATION_DESIGN.md)
- [IMPLEMENTATION_REVIEW.md](/Users/zhoutong/code/verl/sdpo-megatron/IMPLEMENTATION_REVIEW.md)
- [GLM45_AIR_PHYSICS_SECTION3_EXPERIMENT_LOG.md](/Users/zhoutong/code/verl/sdpo-megatron/experiment-logs/GLM45_AIR_PHYSICS_SECTION3_EXPERIMENT_LOG.md)
- [QWEN3_8B_PHYSICS_SECTION3_EXPERIMENT_LOG.md](/Users/zhoutong/code/verl/sdpo-megatron/experiment-logs/QWEN3_8B_PHYSICS_SECTION3_EXPERIMENT_LOG.md)
