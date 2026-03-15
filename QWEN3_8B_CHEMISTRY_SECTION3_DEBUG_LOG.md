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
- Chemistry launcher changes were pushed to `codex/sdpo-megatron-v070` in commit `702ffe84`.
- The first live run is the on-policy `grpo_fsdp` baseline on cluster `verl-qwen3-chemistry-grpo`.
- The first launch attempt hit a SkyPilot resource mismatch because `model-eval` was already up with a different image, so the cluster was recreated before relaunch.
- The recreated `model-eval` cluster successfully reached setup and started job `1`.
- Job `1` failed in trainer config resolution because the chemistry trainer YAMLs expected `CHEMISTRY_TRAIN_FILE` / `CHEMISTRY_VAL_FILE` in the environment.
- The trainer configs have now been patched to derive parquet paths directly from `CHEMISTRY_DATA_DIR`, which is already exported by the SkyPilot launcher.
- The parquet-path fix was pushed to `codex/sdpo-megatron-v070` in commit `ca9f631a`.
- A fresh-cluster relaunch on `verl-qwen3-chemistry-grpo` is now past provisioning, with setup detached and job `1` started.
- The fresh-cluster relaunch got through setup, Ray startup, and trainer config validation before failing in custom reward-function loading.
- The chemistry trainer configs have now been patched to use `${oc.env:VERL_REPO_DIR}/verl/utils/reward_score/feedback/__init__.py` for the reward function path.
- The staged remote model path is `/hai/zhoutong/section3_chemistry_assets/models/Qwen3-8B-Base`, backed by the existing cached checkpoint under `/hai/zhoutong/.modelscope_cache/models/Qwen/Qwen3-8B-Base`.
- The staged remote dataset path is `/hai/zhoutong/section3_chemistry_assets/data/sciknoweval_chemistry`.
- Python syntax and YAML parsing checks passed locally.
- Full Hydra config rendering has not been validated locally because the desktop Python env is missing `packaging`.
- Current focus is pushing the config fix and relaunching the first GRPO baseline on a fresh cluster name so `model-eval` can remain available.

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
- The live step in progress is launching the on-policy GRPO FSDP baseline from the updated SkyPilot path.

### [Resolved] Existing `model-eval` cluster image mismatch blocked the first launch

- The first `sky launch -c model-eval ...` attempt failed with `sky.exceptions.ResourcesMismatchError` because the existing `model-eval` cluster was running a different image (`docker:nvcr.io/nvidia/nemo:25.09.02`) than the chemistry YAML expects.
- Since the staged Chemistry assets live on `/hai`, it was safe to recreate the cluster.
- `sky down -y model-eval` completed successfully, and the chemistry relaunch is now running against a fresh `model-eval` cluster.

### [Resolved] First GRPO launch failed because chemistry parquet env vars were resolved too early

- The recreated `model-eval` cluster reached setup, cloned the branch, generated the chemistry parquet files from the staged JSON data, and started job `1`.
- The run then failed in Hydra/OmegaConf resolution with:
  `InterpolationResolutionError: Environment variable 'CHEMISTRY_TRAIN_FILE' not found`
- Root cause: the chemistry trainer YAMLs referenced `CHEMISTRY_TRAIN_FILE` and `CHEMISTRY_VAL_FILE`, but those were only set inside the shell wrapper and were not guaranteed to exist in the environment seen during trainer config resolution.
- Fix: update the chemistry trainer YAMLs to use `${oc.env:CHEMISTRY_DATA_DIR}/train.parquet` and `${oc.env:CHEMISTRY_DATA_DIR}/test.parquet` directly.

### [WIP] Local Hydra render is blocked by desktop env dependencies

- `python3 -m verl.trainer.main_ppo --cfg job` currently fails locally with `ModuleNotFoundError: No module named 'packaging'`.
- This is a local environment issue on the desktop Python, not a syntax error in the branch.

### [WIP] Relaunch the first GRPO baseline on a fresh cluster name

- Per the current execution preference, the next launch should use a new cluster name instead of reusing `model-eval`.
- The `CHEMISTRY_DATA_DIR`-based config fix has been committed and pushed.
- Fresh-cluster relaunch target: `verl-qwen3-chemistry-grpo`.
- Current state: cluster is up, and the next relaunch on `verl-qwen3-chemistry-grpo` should pick up the absolute reward-path fix.

### [Resolved] Fresh-cluster GRPO run reached reward setup before failing on a repo-relative custom reward path

- The first run on `verl-qwen3-chemistry-grpo` got past the earlier `CHEMISTRY_TRAIN_FILE` interpolation issue.
- It then failed with:
  `FileNotFoundError: Custom module file not found: module_path='verl/utils/reward_score/feedback/__init__.py'`
- Root cause: the reward loader expects a real filesystem path, not a repo-relative string.
- Fix: update all three chemistry trainer configs to use
  `${oc.env:VERL_REPO_DIR}/verl/utils/reward_score/feedback/__init__.py`.
