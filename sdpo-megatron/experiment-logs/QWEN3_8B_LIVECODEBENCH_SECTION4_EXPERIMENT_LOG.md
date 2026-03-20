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

## Current Status Snapshot (2026-03-20)

- The Section 4 code path has been added locally, committed, and pushed.
- Current branch:
  - `codex/sdpo-megatron-v070`
- Pushed commit used for this run:
  - `f9a9b043`
  - message: `Add Qwen3 Section 4 LiveCodeBench launcher`
- The current launcher now follows the same successful Section 3 pattern:
  - remote cluster clones `https://github.com/zhoutong-hai/verl.git`
  - branch `codex/sdpo-megatron-v070`
  - repo path `/root/verl-section4-livecodebench`
- Current live cluster:
  - `verl-qwen3-section4-livecodebench`
- Current SkyPilot status:
  - `INIT`
- Most recent observation:
  - pod came up on Kubernetes
  - cluster has not yet transitioned to `UP`

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
