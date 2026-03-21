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

## Current Status Snapshot (2026-03-21 01:36 PDT)

- The current live Section 4 SDPO run is:
  - cluster `verl-qwen3-section4-livecodebench`
  - job id `150`
  - W&B run [`aev2yvmk`](https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/aev2yvmk)
  - queue status `RUNNING`
- The current launcher path is verified through:
  - cluster launch
  - remote repo clone / install
  - shared-data bootstrap on `/hai`
  - Ray startup
  - FSDP actor/ref + vLLM initialization on 4 GPUs
  - W&B run creation
  - rollout generation
  - actor update
  - first real SDPO step
- Latest observed trainer progress:
  - `training/global_step: 1`
  - `Training Progress:   0%|          | 0/120` followed by the first full step log line
- Latest observed step-1 metrics from job `150`:
  - `self_distillation/feedback_available_fraction: 0.93359375`
  - `self_distillation/feedback_used_fraction: 0.75`
  - `self_distillation/reprompt_sample_fraction: 0.984375`
  - `critic/score/mean: 0.06640625`
  - `response_length/mean: 999.046875`
  - `timing_s/gen: 56.0905`
  - `timing_s/reward: 522.9072`
  - `timing_s/update_actor: 72.7139`
  - `timing_s/step: 660.8877`
  - `perf/throughput: 151.29`
- Current maturity assessment:
  - this relaunch is back in a healthy end-to-end bring-up state
  - it has cleared the first-step correctness bar on the corrected 4-GPU setup
  - it is still pre-validation and should not yet be treated as a completed Section 4 result
- Important hardware note:
  - the cluster shape is still `H200:8`
  - the current Section 4 launcher is intentionally back on 4 GPUs to match the upstream setup and avoid the silent pre-step stall seen on the 8-GPU attempt

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
  - [`3afxjxj1`](https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/3afxjxj1)
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
  - W&B run [`r7co6lpk`](https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/r7co6lpk)
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

### [Resolved] Future launcher defaults now match the allocated node shape better

- The original Section 4 launcher reserved `H200:8` but only exposed GPUs `0,1,2,3` and set `N_GPUS_PER_NODE=4`.
- For subsequent launches, the setup has been updated to:
  - use all 8 visible GPUs by default:
    - `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - `N_GPUS_PER_NODE=8`
  - keep the current live run unchanged while fixing the default for future relaunches

### [Resolved] Future launcher defaults are more conservative before first validation

- To reduce pre-validation cost/risk on new relaunches, the default Section 4 context limits were tightened:
  - `MAX_RESPONSE_LENGTH=4096`
  - `MAX_MODEL_LEN=10240`
  - `MAX_REPROMPT_LEN=6144`
- These values are now environment-driven so they can still be overridden back to longer paper-faithful settings if needed.

### [Note] Previous job `21` was manually interrupted during investigation

- While probing the old run for a Python stack, a manual signal terminated job `21`.
- That failure was investigation-induced, not a trainer-side crash.
- Because of that, the next check was done by relaunching on the same cluster instead of treating job `21` as a clean product failure.

### [Resolved] Same-cluster relaunch `62` reached `training/global_step: 1`

- Relaunched on the same cluster:
  - cluster `verl-qwen3-section4-livecodebench`
  - job id `62`
- W&B run:
  - [`4e7m3pym`](https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/4e7m3pym)
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

### [Resolved] Forward progress beyond step 1 is confirmed

- Additional monitoring after the first step showed that the run continued rather than stalling.
- The trainer advanced to:
  - `training/global_step: 2`
  - `Training Progress:   2%|▏         | 2/120`
- Step-2 metrics:
  - `self_distillation/feedback_available_fraction: 0.9609375`
  - `self_distillation/feedback_used_fraction: 0.8125`
  - `self_distillation/reprompt_sample_fraction: 0.984375`
  - `critic/score/mean: 0.0390625`
  - `response_length/mean: 1860.8046875`
  - `timing_s/gen: 98.5985`
  - `timing_s/reward: 324.1183`
  - `timing_s/update_actor: 71.9843`
  - `timing_s/step: 504.7002`
  - `perf/throughput: 307.77`
- Updated interpretation:
  - the job is in a good multi-step bring-up state now
  - it is making multi-step training progress on the same cluster
  - it is still too early to claim validation success or experiment maturity

### [Resolved] 8-GPU relaunch `102` silently stalled before the first step

- After the launcher was changed to use all 8 visible GPUs by default, the same-cluster relaunch was:
  - cluster `verl-qwen3-section4-livecodebench`
  - job id `102`
  - W&B run [`xrar3y2s`](https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/xrar3y2s)
- What was verified on-cluster:
  - the remote config really did switch to:
    - `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - `N_GPUS_PER_NODE=8`
  - 8 actor init workers came up
  - 8 vLLM servers came up
  - rollout generation completed
- Failure pattern:
  - `run.log` stopped advancing after the initial `Training Progress:   0%|          | 0/120`
  - W&B never reported `training/global_step`
  - the queue stayed `RUNNING`, but the run never emitted the first real step
- Interpretation:
  - this looked like a silent post-generation / pre-step stall on the 8-GPU path
  - it was not a clean crash, but it also did not satisfy the bar for a healthy run

### [Resolved] Launcher was reverted to the upstream 4-GPU Section 4 shape

- The copied upstream Section 4 script in `../SDPO` uses:
  - `GPUS_PER_NODE=4`
- Since the 8-GPU attempt stalled and the known-good earlier runs were also 4-GPU, the launcher was corrected back to:
  - `N_GPUS_PER_NODE=4`
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`
- The tighter context limits were kept:
  - `MAX_RESPONSE_LENGTH=4096`
  - `MAX_MODEL_LEN=10240`
  - `MAX_REPROMPT_LEN=6144`
- Commit:
  - `dc47b47d` `Revert Section 4 launcher to 4 GPUs`

### [Resolved] Same-cluster relaunch `150` reached the first real SDPO step

- Relaunched on the same cluster after the 4-GPU revert:
  - cluster `verl-qwen3-section4-livecodebench`
  - job id `150`
- W&B run:
  - [`aev2yvmk`](https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/aev2yvmk)
- Verified on-cluster before the first step:
  - only 4 `WorkerDict.actor_rollout_ref_init_model` workers were active
  - only 4 vLLM servers were launched
  - the config dump showed:
    - `max_prompt_length: 2048`
    - `max_response_length: 4096`
    - `max_model_len: 10240`
    - `max_reprompt_len: 6144`
    - effective trainer `n_gpus_per_node: 4`
- Additional live-health signal before the first metric flush:
  - the run progressed into `ray::WorkerDict.actor_rollout_ref_update_actor`
  - this confirmed the job had moved beyond rollout generation and into the real actor update path
- First-step metrics from job `150`:
  - `training/global_step: 1`
  - `self_distillation/feedback_available_fraction: 0.93359375`
  - `self_distillation/feedback_used_fraction: 0.75`
  - `self_distillation/reprompt_sample_fraction: 0.984375`
  - `critic/score/mean: 0.06640625`
  - `response_length/mean: 999.046875`
  - `timing_s/gen: 56.0905`
  - `timing_s/reward: 522.9072`
  - `timing_s/update_actor: 72.7139`
  - `timing_s/step: 660.8877`
  - `perf/throughput: 151.29`
- Current interpretation:
  - the corrected 4-GPU relaunch is back in a good running state
  - the 8-GPU regression was avoided without giving up the tighter context defaults

### [Resolved] Live Section 4 run advanced to step `29`, exposing a rich-feedback evaluator bug

- Newer relaunches on the same cluster progressed well past the earlier step-1/step-5 bring-up bar:
  - latest verified Sky job before this handoff:
    - cluster `verl-qwen3-section4-livecodebench`
    - job id `176`
    - queue status `SUCCEEDED`
- The live tail reached:
  - `training/global_step: 29`
  - `self_distillation/feedback_available_fraction: 0.9609375`
  - `self_distillation/feedback_used_fraction: 0.8125`
  - `response_length/mean: 1358.87109375`
  - `timing_s/reward: 885.9562`
  - `timing_s/step: 1007.3362`
- The concrete failure surfaced inside the rich-feedback reward worker:
  - `MemoryError` at `send_conn.send((results, outputs))` in [code.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/code.py)
- Root-cause review against the local upstream `SDPO` checkout showed two real drifts in the copied evaluator:
  - the local port had simplified sparse evaluation into a single subprocess returning `(results, outputs)` for all tests, which could build a very large payload before any truncation
  - the local sparse path also stopped after the first failed test, which weakened the per-test environment-feedback signal compared with the upstream Section 4 implementation
- Corrective change taken:
  - restore the upstream per-test record flow and LeetCode-style feedback formatting in [code.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/code.py)
  - keep the local stdout-capture fixes on top
  - additionally truncate `actual` / `debug` payloads before sending them through the multiprocessing pipe
- Rationale:
  - this keeps the paper-faithful rich-feedback semantics
  - and also removes the transport-side OOM that was corrupting long feedback cases
- Next action after this fix:
  - commit and push the evaluator patch
  - relaunch the same Section 4 SDPO experiment on the existing cluster
  - re-check first-step and multi-step progress on the corrected rich-feedback path

### [In Progress] Corrected rich-feedback rerun is live on the fixed evaluator commit

- Fix commit:
  - `961b0e4b` `Fix Section 4 rich feedback evaluator`
- Because Sky task submission stayed inconsistent on the existing single-node cluster, the rerun was started manually on the live head pod after fast-forwarding the remote checkout to the fixed commit.
- Current corrected rerun:
  - experiment `qwen3_8b_section4_livecodebench_sdpo_fsdp_fixed_feedback_20260321_173144`
  - W&B run [`3ps8pj3z`](https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/3ps8pj3z)
  - log `/root/sky_logs/manual-qwen3-section4-fixed-feedback-20260321_173144.log`
- Verified so far on the corrected path:
  - remote checkout is on commit `961b0e4b`
  - W&B run creation succeeded
  - trainer connected to the live Ray cluster
  - FSDP actor/ref workers loaded
  - vLLM servers initialized
  - `Training Progress:   0%|          | 0/120` was emitted on the corrected run
- Current monitoring goal:
  - wait for the first real `training/global_step`
  - specifically confirm that the previous reward-worker `MemoryError` at `send_conn.send(...)` does not recur once the reward loop begins

### [Update] Corrected rich-feedback rerun cleared the old crash, but the first `10` steps are not yet convincing

- The corrected rerun no longer reproduces the old reward-worker crash:
  - no `MemoryError`
  - no `Traceback`
  - no early process exit through the first `9` completed training steps
- Live run under review:
  - experiment `qwen3_8b_section4_livecodebench_sdpo_fsdp_fixed_feedback_20260321_173144`
  - W&B run [`3ps8pj3z`](https://wandb.ai/hippocraticai/qwen3_8b_section4_livecodebench/runs/3ps8pj3z)
  - log `/root/sky_logs/manual-qwen3-section4-fixed-feedback-20260321_173144.log`
- First-`9`-step behavior:
  - `feedback_available_fraction` stayed high: roughly `0.91 -> 0.97`
  - `feedback_used_fraction` stayed high: roughly `0.69 -> 0.84`
  - `reprompt_sample_fraction` stayed very high: roughly `0.97 -> 1.0`
  - `success_group_fraction` oscillated in a weak band: `0.15625 -> 0.3125`
  - `critic/score/mean` stayed low: `0.03125 -> 0.08984`
  - `response_length/mean` stayed large: about `1073 -> 1557`, later `1269 -> 1327`
  - `response_length/clip_ratio` stayed non-trivial: about `0.066 -> 0.148`, later `0.078 -> 0.090`
  - `teacher_preferred_token_fraction` stayed below `0.5`: roughly `0.428 -> 0.471`
- Best-looking early steps:
  - step `6`
    - `success_group_fraction = 0.28125`
    - `critic/score/mean = 0.08984375`
    - `response_length/clip_ratio = 0.09765625`
  - step `9`
    - `success_group_fraction = 0.3125`
    - `critic/score/mean = 0.07421875`
    - `response_length/clip_ratio = 0.08984375`
- Current concern:
  - the run appears to enter step-`10` validation, but has not yet emitted the step-`10` metric line
  - `validation generation end` is present in the log, yet the trainer still sits on step `9` with no post-validation metric flush
  - so the run is no longer crashing, but it may be stalling or at least progressing very slowly at the first validation boundary
- Current conclusion:
  - this is a meaningful correctness improvement over the previous broken evaluator path
  - however, it is **not yet a convincing reproduction of the Section 4 paper result**
  - reason:
    - the rich-feedback path now runs, but early learning remains weak and noisy
    - response lengths are still too large
    - the step-`10` validation boundary is not yet cleanly passing
- Best next debugging direction:
  - investigate the step-`10` validation stall / flush behavior first
  - then tighten the experiment profile for a cheaper and more stable early signal:
    - shorter response / reprompt budgets
    - less console-heavy validation logging
    - possibly stricter validation generation limits before rejudging Section 4 quality
