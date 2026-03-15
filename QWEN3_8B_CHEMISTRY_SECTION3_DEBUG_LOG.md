# Qwen3-8B Chemistry Section 3 Debug Log

## Goal

Reproduce the SDPO paper's "Train-time RL without rich feedback" setup on the SciKnowEval Chemistry task with `Qwen/Qwen3-8B`, then compare:

1. on-policy GRPO baseline
2. upstream-style non-Megatron SDPO
3. Megatron SDPO port on this branch

## Original Task

Figure out how to reproduce the paper setup for train-time RL without rich feedback with `Qwen3-8B`, include the original non-Megatron SDPO implementation in this branch, and start with a Chemistry pilot before scaling wider.

## Working Plan

1. Integrate the upstream non-Megatron SDPO path into this branch.
2. Add Chemistry-specific configs for GRPO FSDP, upstream-style SDPO FSDP, and Megatron SDPO.
3. Add a local preprocessor and run script so the Chemistry pilot is runnable from this repo.
4. Launch and debug the Chemistry pilot one job at a time.
5. Record metrics, blockers, and resolution history here.

## Commands

### Prepare or regenerate Chemistry parquet

```bash
python3 /Users/zhoutong/code/verl/examples/data_preprocess/sdpo_generalization.py \
  --data_source /Users/zhoutong/code/SDPO/datasets/sciknoweval/chemistry
```

### GRPO FSDP baseline

```bash
VARIANT=grpo_fsdp \
bash /Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen3_8b_sciknoweval_chemistry_section3.sh
```

### SDPO FSDP original-style variant

```bash
VARIANT=sdpo_fsdp \
bash /Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen3_8b_sciknoweval_chemistry_section3.sh
```

### SDPO Megatron variant

```bash
VARIANT=sdpo_megatron \
bash /Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen3_8b_sciknoweval_chemistry_section3.sh
```

## Monitoring Metrics

- `val-core/sciknoweval/acc/mean@16`
- `val-aux/sciknoweval/reward/mean@16`
- `self_distillation/reprompt_sample_fraction`
- `self_distillation/active_token_fraction`
- `self_distillation/active_sample_fraction`
- `self_distillation/teacher_preferred_token_fraction`
- `self_distillation/empty_target_batch`
- `actor/pg_loss`
- `actor/grad_norm`
- `actor/entropy`
- `perf/throughput`
- `perf/time_per_step`

## Current Status Snapshot

- Upstream-style FSDP SDPO actor path has been integrated into this branch.
- Trainer now supports choosing actor-side teacher scoring for the original FSDP SDPO path.
- Chemistry-specific configs and a single run script have been added.
- Chemistry assets are staged on `model-eval` under `/hai/zhoutong/section3_chemistry_assets/`.
- A SkyPilot launcher has been updated to use the staged `/hai/zhoutong` model and dataset paths directly.
- The staged remote model path is `/hai/zhoutong/section3_chemistry_assets/models/Qwen3-8B-Base`, backed by the existing cached checkpoint under `/hai/zhoutong/.modelscope_cache/models/Qwen/Qwen3-8B-Base`.
- The staged remote dataset path is `/hai/zhoutong/section3_chemistry_assets/data/sciknoweval_chemistry`.
- Python syntax and YAML parsing checks passed locally.
- Full Hydra config rendering has not been validated locally because the desktop Python env is missing `packaging`.
- Next launch target is the on-policy `grpo_fsdp` baseline on the updated SkyPilot path.

## Debug Notes

### [Resolved] Brought the original non-Megatron SDPO path into this branch

- Replaced [dp_actor.py](/Users/zhoutong/code/verl/verl/workers/actor/dp_actor.py) with the upstream SDPO FSDP actor implementation.
- Updated [fsdp_workers.py](/Users/zhoutong/code/verl/verl/workers/fsdp_workers.py) so the actor can attach the reference model as the SDPO teacher and consume the new `compute_log_prob()` output structure.

### [Resolved] Added a switch for teacher scoring mode

- Added `self_distillation.teacher_scoring_mode` in [actor.py](/Users/zhoutong/code/verl/verl/workers/config/actor.py).
- `trainer_ref` keeps the current Megatron path.
- `actor_worker` enables the upstream-style FSDP SDPO path.

### [Resolved] Added reward support for the Section 3 task family

- Added [feedback/__init__.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/__init__.py), [mcq.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/mcq.py), and [tooluse.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/tooluse.py).
- This is enough for SciKnowEval and ToolUse, which are the no-rich-feedback tasks from Section 3.

### [Resolved] Added Chemistry pilot configs and runner

- Added [grpo_fsdp_sciknoweval_chemistry_trainer.yaml](/Users/zhoutong/code/verl/verl/trainer/config/grpo_fsdp_sciknoweval_chemistry_trainer.yaml).
- Added [sdpo_fsdp_sciknoweval_chemistry_trainer.yaml](/Users/zhoutong/code/verl/verl/trainer/config/sdpo_fsdp_sciknoweval_chemistry_trainer.yaml).
- Added [sdpo_megatron_sciknoweval_chemistry_trainer.yaml](/Users/zhoutong/code/verl/verl/trainer/config/sdpo_megatron_sciknoweval_chemistry_trainer.yaml).
- Added [run_qwen3_8b_sciknoweval_chemistry_section3.sh](/Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen3_8b_sciknoweval_chemistry_section3.sh).

### [Resolved] Added a single-node SkyPilot launcher for the Chemistry pilot

- Added [verl-qwen3-section3-chemistry.yaml](/Users/zhoutong/code/verl/examples/skypilot/verl-qwen3-section3-chemistry.yaml).
- The launcher now uses the pre-staged remote assets under `/hai/zhoutong/section3_chemistry_assets/`, clones this fork branch, and can switch among `grpo_fsdp`, `sdpo_fsdp`, and `sdpo_megatron` through `envs.VARIANT`.

### [Resolved] Staged the Chemistry model and dataset on `model-eval`

- Staged `train.json` and `test.json` under `/hai/zhoutong/section3_chemistry_assets/data/sciknoweval_chemistry`.
- Verified the existing cached Qwen3 base checkpoint under `/hai/zhoutong/.modelscope_cache/models/Qwen/Qwen3-8B-Base`.
- Exposed a stable experiment model path at `/hai/zhoutong/section3_chemistry_assets/models/Qwen3-8B-Base` via symlink, so the launch no longer depends on a runtime Hugging Face download.

### [WIP] Validate config composition and launch the first Chemistry run

- Local script syntax and SkyPilot YAML parsing passed.
- The next live step is launching the on-policy GRPO FSDP baseline from the updated SkyPilot path.

### [WIP] Local Hydra render is blocked by desktop env dependencies

- `python3 -m verl.trainer.main_ppo --cfg job` currently fails locally with `ModuleNotFoundError: No module named 'packaging'`.
- This is a local environment issue on the desktop Python, not a syntax error in the branch.
