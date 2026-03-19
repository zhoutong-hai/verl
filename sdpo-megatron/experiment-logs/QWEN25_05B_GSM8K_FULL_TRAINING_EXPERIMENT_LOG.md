# Qwen2.5-0.5B GSM8K Full Training Debug Log

Last updated: 2026-03-15

## Goal

Run full GSM8K Megatron training with Qwen2.5-0.5B-Instruct on the working SkyPilot path,
track SDPO-specific behavior during training, then run a matched GRPO baseline for comparison.

## Original Task

- Move beyond the smoke test to a full GSM8K training job.
- Evaluate performance on the validation dataset.
- Add metrics that make SDPO behavior and training health easy to monitor.
- Run one SDPO job and one GRPO job with the same setup for comparison.
- Keep a repo-local experiment log, similar to the smoke debug log, with issue history and status snapshots.

## Working Plan

1. Reuse the working smoke-test setup as the base runtime path.
2. Add SDPO-specific monitoring metrics and any missing training-health metrics needed for debugging.
3. Create full-run trainer configs for both SDPO and GRPO on GSM8K.
4. Create full-run launch commands and launchers using the same working SkyPilot + `/hai` asset setup.
5. Launch SDPO first, monitor it through early training and validation, then launch the matched GRPO run.
6. Compare SDPO vs GRPO on validation accuracy, reward, SDPO activation, and training stability.

## Commands

Local direct trainer entrypoint:

```bash
VARIANT=sdpo bash /Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen2_5_0_5b_gsm8k_megatron_full.sh
```

```bash
VARIANT=grpo bash /Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen2_5_0_5b_gsm8k_megatron_full.sh
```

Canonical asset paths:

```bash
TRAIN_FILE=/hai/zhoutong/sdpo_megatron_smoke_qwen25_05b/data/gsm8k/train.parquet
VAL_FILE=/hai/zhoutong/sdpo_megatron_smoke_qwen25_05b/data/gsm8k/test.parquet
MODEL_PATH=/hai/zhoutong/sdpo_megatron_smoke_qwen25_05b/models/Qwen2.5-0.5B-Instruct
```

SkyPilot full-run launch commands:

```bash
export WANDB_API_KEY='***'
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
sky launch -c verl-qwen05b-sdpo-full /Users/zhoutong/code/verl/examples/skypilot/verl-sdpo-megatron-full-qwen05b.yaml --secret WANDB_API_KEY -y
```

```bash
export WANDB_API_KEY='***'
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
sky launch -c verl-qwen05b-grpo-full /Users/zhoutong/code/verl/examples/skypilot/verl-grpo-megatron-full-qwen05b.yaml --secret WANDB_API_KEY -y
```

## Monitoring Metrics

Primary outcome metrics:

- `val-core/openai/gsm8k/acc/mean@1`
- `val-aux/openai/gsm8k/reward/mean@1`

SDPO activation and coverage:

- `self_distillation/reprompt_sample_fraction`
- `self_distillation/success_group_fraction`
- `self_distillation/success_sample_fraction`
- `self_distillation/feedback_available_fraction`
- `self_distillation/feedback_used_fraction`
- `self_distillation/solution_used_fraction`
- `self_distillation/solution_and_feedback_fraction`
- `self_distillation/feedback_only_fraction`
- `self_distillation/empty_target_batch`
- `self_distillation/active_sample_fraction`
- `self_distillation/active_token_fraction`

Teacher-student and EMA diagnostics:

- `self_distillation/student_minus_teacher_logprob_mean`
- `self_distillation/student_minus_teacher_logprob_abs_mean`
- `self_distillation/teacher_preferred_token_fraction`
- `self_distillation/per_token_loss_mean`
- `self_distillation/teacher_prompt_length_mean`
- `self_distillation/teacher_prompt_length_max`
- `self_distillation/teacher_update_rate`
- `self_distillation/teacher_actor_param_rms_before_update`
- `self_distillation/teacher_actor_param_rms_after_update`

Training-health metrics:

- `actor/pg_loss`
- `actor/grad_norm`
- `actor/entropy`
- `actor/lr`
- `training/global_step`
- `rollout_corr/kl`
- `rollout_corr/log_ppl_diff`
- `training/rollout_actor_probs_pearson_corr`
- `training/rollout_probs_diff_mean`
- `training/rollout_probs_diff_max`
- `critic/rewards/mean`
- `critic/advantages/mean`
- `response_length/mean`
- `response_length/clip_ratio`
- `response/aborted_ratio`
- `perf/throughput`
- `perf/time_per_step`
- `perf/max_memory_allocated_gb`

## Current Status Snapshot

Update this section every time a task completes, a new issue is found, or an old issue is resolved.

- Last checked: 2026-03-15
- Scope: full GSM8K training setup for Qwen2.5-0.5B, with SDPO vs GRPO comparison
- Branch / last pushed commit: `codex/sdpo-megatron-v070` at `12d107cc`
- Current run stage reached:
  pre-launch setup is complete locally; full-run configs, instrumentation, and SkyPilot launchers have been added and validated
- Current active blocker:
  no runtime blocker identified yet; the next step is to commit/push the new repo code so the remote clone-based run picks it up
- Plan status:
  1. Reuse the working smoke-test setup as the base runtime path. `completed`
  2. Add SDPO-specific monitoring metrics and any missing training-health metrics needed for debugging. `completed`
  3. Create full-run trainer configs for both SDPO and GRPO on GSM8K. `completed`
  4. Create full-run launch commands and launchers using the same working SkyPilot + `/hai` asset setup. `completed`
  5. Launch SDPO first, monitor it through early training and validation, then launch the matched GRPO run. `pending`
  6. Compare SDPO vs GRPO on validation accuracy, reward, SDPO activation, and training stability. `pending`
- Recent completed milestones:
  - the smoke test completed successfully on task `9`
  - the smoke path already validated SkyPilot, `/hai` mount, custom Ray, Megatron actor/ref init, and SDPO loss execution
  - full-run trainer configs have been created locally for SDPO and GRPO
  - a full-run local launcher script has been created locally
  - dedicated SkyPilot full-run launchers have been created locally for SDPO and GRPO
  - the new Python metrics and full-run configs passed local validation
- Local workspace state:
  - local changes are in progress for full-run setup and instrumentation

## Debug Notes

### [Resolved] 2026-03-15: Smoke path validated before moving to full training

Why this matters:
- the smoke run already proved that the SDPO Megatron path can execute end to end
- full-training work can now focus on scaling the same setup rather than debugging the original port from scratch

Status:
- smoke test completed successfully on task `9`
- see `/Users/zhoutong/code/verl/sdpo-megatron/experiment-logs/SDPO_SMOKE_EXPERIMENT_LOG.md` for the detailed issue history

### [WIP] 2026-03-15: Full-run monitoring surface expansion

Objective:
- add the missing metrics that make full SDPO training easier to interpret and compare against GRPO

Current work:
- add SDPO target-source metrics in the trainer
- add active-token and teacher-student gap metrics in the SDPO loss
- add EMA teacher drift metrics in the Megatron worker update path

Status:
- local code changes are complete
- local validation passed
- no longer the active blocker

### [Resolved] 2026-03-15: Full-run launcher setup

Objective:
- create a full-run launcher that mirrors the smoke setup closely enough to reuse the known-good runtime path

Current work:
- full-run trainer configs and local script created
- SkyPilot full-run launcher(s) still need to be added

Status:
- dedicated SDPO and GRPO SkyPilot YAMLs have been created locally
- exact launch commands are now recorded above
- no longer the active blocker

### [WIP] 2026-03-15: First full SDPO launch

Objective:
- push the new repo code/config to the fork branch
- launch the SDPO full run on the validated SkyPilot path
- monitor the early training and validation behavior before moving on to GRPO

Status:
- launch not started yet from this log
- next step is commit, push, and launch SDPO
