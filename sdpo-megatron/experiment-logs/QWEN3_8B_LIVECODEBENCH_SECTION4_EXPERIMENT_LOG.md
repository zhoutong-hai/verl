# Qwen3-8B LiveCodeBench Section 4 Experiment Log

## Goal

Run the Section 4 rich-feedback coding experiment on Qwen3 8B using LiveCodeBench v6, following the same remote-launch pattern that worked for the earlier Section 3 SkyPilot runs.

## Original Task

Start a paper-faithful Section 4 path with minimal code change by copying the upstream SDPO rich-feedback code flow into the current `verl` repo, then launch it on `hai-training` with SkyPilot.

## Working Plan

1. Copy the upstream rich-feedback code evaluator into this repo.
2. Add LiveCodeBench v6 data-prep utilities that write into shared `/hai` storage.
3. Add Section 4 FSDP trainer configs for:
   - `sdpo_fsdp`
   - `grpo_fsdp`
4. Add a dedicated Section 4 run script and SkyPilot launcher.
5. Commit and push the changes before launching, because the cluster clones the fork branch from Git.
6. Launch the SDPO Section 4 run on `hai-training`.

## Commands

```bash
source ~/.zshrc
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate

sky down -y verl-qwen3-section4-livecodebench

sky launch -c verl-qwen3-section4-livecodebench \
  /Users/zhoutong/code/verl/examples/skypilot/verl-qwen3-section4-livecodebench.yaml \
  --env VARIANT=sdpo_fsdp \
  --secret WANDB_API_KEY \
  -d -y
```

## Runtime Assets

- Cluster launcher:
  - [verl-qwen3-section4-livecodebench.yaml](/Users/zhoutong/code/verl/examples/skypilot/verl-qwen3-section4-livecodebench.yaml)
- Run script:
  - [run_qwen3_8b_livecodebench_section4.sh](/Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen3_8b_livecodebench_section4.sh)
- Rich-feedback reward entrypoint:
  - [__init__.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/__init__.py)
- Code environment evaluator:
  - [code.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/code.py)
- Data prep:
  - [load_livecodebench_v6.py](/Users/zhoutong/code/verl/examples/data_preprocess/load_livecodebench_v6.py)
  - [split_livecodebench_tests.py](/Users/zhoutong/code/verl/examples/data_preprocess/split_livecodebench_tests.py)

## Dataset / Paths

- Shared dataset directory:
  - `/hai/zhoutong/section4_livecodebench_assets/data/lcb_v6`
- Shared validation dump root:
  - `/hai/zhoutong/section4_livecodebench_assets/validation_generations`
- Model path:
  - `/hai/zhoutong/section3_chemistry_assets/models/Qwen3-8B-Base`

## Monitoring Metrics

- `val-core/livecodebench/acc/mean@4`
- `val-core/livecodebench/acc/best@4/mean`
- `self_distillation/success_group_fraction`
- `self_distillation/reprompt_sample_fraction`
- `self_distillation/feedback_used_fraction`
- `self_distillation/empty_target_batch`
- `critic/score/mean`
- `response_length/mean`
- inline validation samples in `run.log`
- validation dumps under `/hai/zhoutong/section4_livecodebench_assets/validation_generations/`

## Current Status Snapshot (2026-03-21 00:18 PT)

- The Section 4 SDPO path is working end-to-end through:
  - cluster launch
  - remote repo clone / install
  - shared-data bootstrap on `/hai`
  - Ray startup
  - FSDP actor/ref + vLLM initialization
  - W&B run creation
  - first real SDPO training step on the unpatched functional-feedback log path
- Current branch:
  - `codex/sdpo-megatron-v070`
- Current live cluster:
  - `verl-qwen3-section4-livecodebench`
- Latest pushed fix commit:
  - `6cd62483`
  - message: `Capture functional feedback stdout`
- Current relaunch using `6cd62483`:
  - job id: `21`
  - W&B run: `https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/r7co6lpk`
  - status: running
  - confirmed on this relaunch:
    - first real SDPO step reached: `training/global_step: 1`
    - no leaked standalone `True`
    - no leaked standalone `False`
    - no leaked standalone digit-only lines
    - no leaked `Test case` lines
    - no leaked `Invalid add value!` / `Exceeds maximum capacity!` lines
  - current watch item:
    - first-step reward time on this relaunch was slower than the previous run, so latency is being watched even though functionality is correct

## Debug Notes

### [Resolved] First Section 4 launcher attempt used local workdir sync instead of the established Git-clone pattern

- The earlier Section 3 Qwen Physics log records a real failure caused by unpushed local changes:
  - the cluster cloned the older remote branch state
  - fix there was to commit and push before relaunching
- Because of that, the Section 4 launcher was changed back to the same pattern as the Section 3 launchers:
  - clone fork branch in `setup:`
  - install from the cloned remote repo
  - launch from that remote checkout

### [Resolved] Section 4 branch content is now pushed for remote use

- Section 4 files were committed and pushed to:
  - branch `codex/sdpo-megatron-v070`
  - remote `fork`
- This makes the new launcher/config/reward/data scripts visible to the remote SkyPilot job.

### [In Progress] First proper Git-backed Section 4 launch is underway

- The currently running launch uses:
  - pushed commit `f9a9b043`
  - launcher [verl-qwen3-section4-livecodebench.yaml](/Users/zhoutong/code/verl/examples/skypilot/verl-qwen3-section4-livecodebench.yaml)
  - `VARIANT=sdpo_fsdp`
- The next meaningful checkpoint is:
  - cluster reaches `UP`
  - remote `setup:` completes
  - dataset bootstrap begins if `/hai/zhoutong/section4_livecodebench_assets/data/lcb_v6` is missing files
  - `main_ppo` starts and emits the first training/validation logs

### [Resolved] Shared LiveCodeBench v6 dataset bootstrap succeeded

- The launcher created the shared Section 4 dataset assets on `/hai`:
  - `lcb_v6.json`
  - `train.json`
  - `test.json`
  - `train.parquet`
  - `test.parquet`
- Validation dump directories were also created successfully under:
  - `/hai/zhoutong/section4_livecodebench_assets/validation_generations/`

### [Resolved] First full SDPO Section 4 run reached `training/global_step: 1`

- The first successful end-to-end training run on this launcher reached the first real SDPO update.
- W&B run:
  - `https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/3afxjxj1`
- Key first-step metrics from that run:
  - `training/global_step: 1`
  - `self_distillation/feedback_available_fraction: 0.9453125`
  - `self_distillation/feedback_used_fraction: 0.8125`
  - `self_distillation/reprompt_sample_fraction: 0.98828125`
  - `critic/score/mean: 0.0546875`
  - `response_length/mean: 1465.08203125`
  - `timing_s/gen: 90.0713`
  - `timing_s/reward: 14.8793`
  - `timing_s/update_actor: 80.4759`
  - `timing_s/step: 195.0009`

### [Resolved] Assert-style feedback log spam was removed

- The earlier rich-feedback harness was leaking assert-test stdout into the trainer log.
- Fix:
  - commit `ae8ee043`
  - route assert-test stdout through the capture buffer in [code.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/code.py)
- Result:
  - no more leaked standalone `True` / `False`
  - no more leaked `Test case ...` lines

### [Resolved] Functional/context top-level stdout leak identified and patched

- After the assert fix, the next run still leaked raw output during the first training step:
  - standalone digit-only lines like `4`, `0`, `3`, `80`
  - raw messages like `Invalid add value!` and `Exceeds maximum capacity!`
- Root cause:
  - `run_test_func()` executed the model completion before redirecting `stdout`
  - shared `context` code in `run_tests_for_one_example()` also executed outside the capture path
- Fix:
  - commit `6cd62483`
  - add `_exec_and_capture_output(...)`
  - use it for:
    - functional completion exec in [code.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/code.py)
    - shared context exec in [code.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/code.py)
- Relaunch procedure:
  - push commit to `fork/codex/sdpo-megatron-v070`
  - fast-forward remote checkout on the same cluster
  - cancel old training job
  - relaunch on the same `verl-qwen3-section4-livecodebench` cluster

### [Resolved] Patched relaunch reached first step without leaked functional/context stdout

- Current patched run:
  - job id `21`
  - W&B run `r7co6lpk`
- Verified so far:
  - remote checkout is on commit `6cd62483`
  - no leaked standalone `True` / `False`
  - no leaked standalone digit-only lines
  - no leaked `Invalid add value!` / `Exceeds maximum capacity!` lines
  - first real SDPO step reached on the patched path
- First-step metrics on the patched relaunch:
  - `training/global_step: 1`
  - `self_distillation/feedback_available_fraction: 0.9375`
  - `self_distillation/feedback_used_fraction: 0.78125`
  - `self_distillation/reprompt_sample_fraction: 0.9921875`
  - `critic/score/mean: 0.0625`
  - `response_length/mean: 1150.3828125`
  - `timing_s/gen: 88.5376`
  - `timing_s/reward: 434.4397`
  - `timing_s/update_actor: 78.3297`
  - `timing_s/step: 610.2697`
- Remaining observation:
  - the functional fix removed the log leakage, but this relaunch had a noticeably slower reward/step time than the previous unpatched run
  - this needs more monitoring to tell whether it is first-batch variance or a real performance regression

### [Resolved] Validation cadence was reduced for the next relaunch

- To make the next live run easier to monitor and reduce unnecessary churn, the launcher was updated to make validation/logging less aggressive:
  - `trainer.test_freq=10`
  - `trainer.log_val_generations=4`
  - `trainer.print_val_generations=1`
- This was kept as a launcher-only change in:
  - [run_qwen3_8b_livecodebench_section4.sh](/Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen3_8b_livecodebench_section4.sh)
- Commit:
  - `c2d50037`

### [Note] Previous job `21` was manually interrupted during investigation

- While probing the old run for a Python stack, a manual signal terminated job `21`.
- That failure was investigation-induced, not a trainer-side crash.
- Because of that, the next check was done by relaunching on the same cluster instead of treating job `21` as a clean product failure.

### [Resolved] Same-cluster relaunch `62` reached `training/global_step: 1`

- Relaunched on the same cluster:
  - cluster `verl-qwen3-section4-livecodebench`
  - job id `62`
- W&B run:
  - `https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/4e7m3pym`
- Verified on-cluster:
  - remote checkout is using the updated launcher settings
  - `main_ppo` is running with:
    - `trainer.test_freq=10`
    - `trainer.log_val_generations=4`
    - `trainer.print_val_generations=1`
  - the run reached the first real SDPO step and advanced the trainer progress bar to:
    - `Training Progress:   1%|          | 1/120`
- First-step metrics from job `62`:
  - `training/global_step: 1`
  - `self_distillation/feedback_available_fraction: 0.9140625`
  - `self_distillation/feedback_used_fraction: 0.71875`
  - `self_distillation/reprompt_sample_fraction: 0.98046875`
  - `critic/score/mean: 0.0859375`
  - `response_length/mean: 1192.44921875`
  - `timing_s/gen: 88.1865`
  - `timing_s/reward: 27.4059`
  - `timing_s/update_actor: 79.6239`
  - `timing_s/step: 204.1362`
  - `perf/throughput: 550.45`

### [Monitoring] Current live state after step 1 looks healthy enough to keep running

- The cluster job is still active:
  - `62  ... RUNNING`
- Post-step monitoring shows:
  - no traceback or fatal error in `run.log`
  - `TaskRunner.run` remains alive
  - a helper `ray::TaskRunner.run` has been consuming CPU after step 1 while the main trainer waits on it
  - no new actor-task failures have appeared in Ray task summaries
- Current interpretation:
  - the run is no longer blocked on bring-up
  - it has cleared the end-to-end correctness bar through the first real update
  - the remaining watch item is forward progress beyond step 1, not a known crash
