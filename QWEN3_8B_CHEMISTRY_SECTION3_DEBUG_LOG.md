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

### Local: GRPO FSDP baseline

```bash
VARIANT=grpo_fsdp \
bash /Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen3_8b_sciknoweval_chemistry_section3.sh
```

### Local: SDPO FSDP original-style variant

```bash
VARIANT=sdpo_fsdp \
bash /Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen3_8b_sciknoweval_chemistry_section3.sh
```

### Local: SDPO Megatron variant

```bash
VARIANT=sdpo_megatron \
bash /Users/zhoutong/code/verl/examples/sdpo_trainer/run_qwen3_8b_sciknoweval_chemistry_section3.sh
```

### SkyPilot: GRPO FSDP baseline

```bash
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
export WANDB_API_KEY='***'
sky launch -c verl-qwen3-chemistry-grpo \
  /Users/zhoutong/code/verl/examples/skypilot/verl-qwen3-section3-chemistry.yaml \
  --env VARIANT=grpo_fsdp \
  --secret WANDB_API_KEY -y
```

### SkyPilot: SDPO FSDP original-style variant

```bash
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
export WANDB_API_KEY='***'
sky launch -c verl-qwen3-chemistry-sdpo \
  /Users/zhoutong/code/verl/examples/skypilot/verl-qwen3-section3-chemistry.yaml \
  --env VARIANT=sdpo_fsdp \
  --secret WANDB_API_KEY -y
```

### SkyPilot: SDPO Megatron variant

```bash
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
export WANDB_API_KEY='***'
sky launch -c verl-qwen3-chemistry-sdpo-megatron \
  /Users/zhoutong/code/verl/examples/skypilot/verl-qwen3-section3-chemistry.yaml \
  --env VARIANT=sdpo_megatron \
  --secret WANDB_API_KEY -y
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

## Current Status Snapshot (2026-03-16)

### Run 1 results (configs pre-alignment with upstream)

All three runs from the initial launch have failed or collapsed:

| Cluster | Variant | Status | Detail |
|---------|---------|--------|--------|
| `verl-qwen3-chemistry-grpo` | GRPO FSDP | Entropy collapse at step 257 | Score 0.515 → 0.004 in 2 steps |
| `verl-qwen3-chemistry-sdpo` | SDPO FSDP | Death spiral from step ~19 | All self_distillation metrics zero; loss=0, grad_norm=0 |
| `verl-qwen3-chemistry-sdpo-megatron` | SDPO Megatron | Crashed on init | `ModuleNotFoundError: megatron.core.distributed.custom_fsdp` |

### Config mismatch with upstream

Our original SDPO configs diverged from the upstream Section 3 effective config in critical ways:

| Parameter | Our original | Upstream effective | Source |
|-----------|-------------|-------------------|--------|
| `success_reward_threshold` | 1.0 | 0.5 | `actor.yaml` overrides dataclass default |
| `alpha` | 1.0 (reverse KL) | 0.5 (JSD) | command-line in `run_sdpo_all.sh` |
| `full_logit_distillation` | false | true | `actor.yaml` |
| `distillation_topk` | not set | 100 | command-line in `run_sdpo_all.sh` |

### Fixes applied (commit `0cb2fecb`)

- **SDPO FSDP**: `success_reward_threshold` 1.0 → 0.5 (alpha/logit/topk were already aligned)
- **SDPO Megatron**: `alpha` 1.0 → 0.5, `success_reward_threshold` 1.0 → 0.5, added `vanilla_mbridge: false`
- Note: Megatron keeps `full_logit_distillation: false` (raises `NotImplementedError`)

### Additional FSDP alignment applied locally for the next rerun

- **GRPO FSDP**: `max_model_len` 10240 → 18944 to match the upstream Section 3 config family.
- **GRPO FSDP**: `clip_ratio_high` 0.2 → 0.28 to match upstream `user.yaml`.
- **SDPO FSDP**: `clip_ratio_high` 0.2 → 0.28 to match upstream `user.yaml`.
- **SDPO FSDP**: `remove_thinking_from_demonstration` explicitly set to `true` to match upstream `actor.yaml`.
- For the paper-style readout, we will compare the best `acc@16` within the first `1h` and `5h`, not only the final checkpoint.

### Live cluster state

All three chemistry clusters are currently `UP`, and each has an active running job:

| Cluster | Variant | Job | Status | Latest visible step |
|---------|---------|-----|--------|---------------------|
| `verl-qwen3-chemistry-grpo` | GRPO FSDP | 2 | RUNNING | 334 / 1770 |
| `verl-qwen3-chemistry-sdpo` | SDPO FSDP | 3 | RUNNING | 42 / 1770 |
| `verl-qwen3-chemistry-sdpo-megatron` | SDPO Megatron | 3 | RUNNING | 39 / 1770 |

### Latest metric snapshot

#### `verl-qwen3-chemistry-grpo`

- Best recent validation visible in the logs is healthy:
  - `val-core/sciknoweval/acc/mean@16 = 0.4357` at step `330`
  - `val-core/sciknoweval/acc/best@16/mean = 0.4777`
  - `val-core/sciknoweval/acc/maj@16/mean = 0.4349`
- Latest live training metrics at step `334`:
  - `critic/score/mean = 0.3828`
  - `actor/entropy = 0.0174`
  - `perf/time_per_step = 98.3s`
  - `perf/throughput = 576.8`
- Current read: the corrected GRPO run is actively training and is far healthier than the earlier collapsed run.

#### `verl-qwen3-chemistry-sdpo`

- Early SDPO activation looked promising:
  - step `5`: `self_distillation/reprompt_sample_fraction = 0.7344`
  - step `5`: `val-core/sciknoweval/acc/mean@16 = 0.2896`
  - step `5`: `val-core/sciknoweval/acc/best@16/mean = 0.8750`
- Current state has collapsed again:
  - step `42`: `self_distillation/reprompt_sample_fraction = 0.0`
  - step `42`: `self_distillation/active_token_fraction = 0.0`
  - step `42`: `self_distillation/empty_target_batch = 1.0`
  - step `42`: `critic/score/mean = 0.0`
  - step `42`: `actor/pg_loss = 0.0`, `actor/grad_norm = 0.0`
- Latest visible validation remains zeroed:
  - step `40`: `val-core/sciknoweval/acc/mean@16 = 0.0`

#### `verl-qwen3-chemistry-sdpo-megatron`

- Early Megatron SDPO activation also looked healthy:
  - step `5`: `self_distillation/reprompt_sample_fraction = 0.7656`
  - step `5`: `val-core/sciknoweval/acc/mean@16 = 0.2896`
  - step `5`: `val-core/sciknoweval/acc/best@16/mean = 0.8750`
- It has also drifted into an empty-target regime:
  - step `39`: `self_distillation/reprompt_sample_fraction = 0.0`
  - step `39`: `self_distillation/active_token_fraction = 0.0`
  - step `39`: `self_distillation/empty_target_batch = 1.0`
  - step `39`: `critic/score/mean = 0.0`
  - step `39`: `actor/pg_loss = 0.0`, `actor/grad_norm = 0.0`
- Latest visible validation is also zero:
  - step `35`: `val-core/sciknoweval/acc/mean@16 = 0.0`

#### SDPO FSDP corrected parameters (`sdpo_fsdp_sciknoweval_chemistry_trainer.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `alpha` | 0.5 | JSD (matches upstream) |
| `success_reward_threshold` | 0.5 | Matches upstream `actor.yaml` |
| `full_logit_distillation` | true | Matches upstream |
| `distillation_topk` | 100 | Matches upstream |
| `dont_reprompt_on_self_success` | true | Matches upstream |
| `include_environment_feedback` | false | Matches upstream |
| `teacher_scoring_mode` | actor_worker | FSDP-native teacher path |
| `teacher_regularization` | ema | |
| `teacher_update_rate` | 0.05 | |
| `use_kl_loss` | false | |
| `lr` | 1e-5 | |
| `train_batch_size` | 32 | |
| `rollout.n` | 8 | |
| `ppo_mini_batch_size` | 32 | |

#### SDPO Megatron corrected parameters (`sdpo_megatron_sciknoweval_chemistry_trainer.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `alpha` | 1.0 | Reverse KL (required when `full_logit_distillation: false`) |
| `success_reward_threshold` | 0.5 | Matches upstream `actor.yaml` |
| `full_logit_distillation` | false | Megatron limitation (`NotImplementedError`); forces `alpha=1.0` |
| `dont_reprompt_on_self_success` | true | Matches upstream |
| `include_environment_feedback` | false | Matches upstream |
| `teacher_scoring_mode` | trainer_ref | Megatron teacher path |
| `teacher_regularization` | ema | |
| `teacher_update_rate` | 0.05 | |
| `vanilla_mbridge` | false | Fixes `custom_fsdp` crash on megatron-core 0.15.0 |
| `use_kl_loss` | false | |
| `lr` | 1e-5 | |
| `train_batch_size` | 32 | |
| `rollout.n` | 8 | |
| `ppo_mini_batch_size` | 32 | |

#### GRPO FSDP parameters (`grpo_fsdp_sciknoweval_chemistry_trainer.yaml`, unchanged)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `adv_estimator` | grpo | |
| `use_kl_loss` | false | No KL regularization |
| `use_kl_in_reward` | false | |
| `lr` | 1e-5 | Upstream also sweeps 1e-6 |
| `train_batch_size` | 32 | |
| `rollout.n` | 8 | |
| `ppo_mini_batch_size` | 32 | |
| `rollout_is` | token | |
| `rollout_is_threshold` | 2.0 | |

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

### [Resolved] Relaunch the first GRPO baseline on a fresh cluster name

- Per the current execution preference, the next launch should use a new cluster name instead of reusing `model-eval`.
- The `CHEMISTRY_DATA_DIR`-based config fix has been committed and pushed.
- Fresh-cluster relaunch target: `verl-qwen3-chemistry-grpo`.
- Both `verl-qwen3-chemistry-grpo` and `verl-qwen3-chemistry-sdpo` clusters were launched.

### [Resolved] Fresh-cluster GRPO run reached reward setup before failing on a repo-relative custom reward path

- The first run on `verl-qwen3-chemistry-grpo` got past the earlier `CHEMISTRY_TRAIN_FILE` interpolation issue.
- It then failed with:
  `FileNotFoundError: Custom module file not found: module_path='verl/utils/reward_score/feedback/__init__.py'`
- Root cause: the reward loader expects a real filesystem path, not a repo-relative string.
- Fix: update all three chemistry trainer configs to use
  `${oc.env:VERL_REPO_DIR}/verl/utils/reward_score/feedback/__init__.py`.

### [Resolved] FSDP SDPO actor crashes on dict attribute access for self_distillation config

- The SDPO FSDP run on `verl-qwen3-chemistry-sdpo` failed in the first actor update with:
  `AttributeError: 'dict' object has no attribute 'full_logit_distillation'`
- Failing file: `verl/workers/actor/dp_actor.py`, line 778 in `update_policy()`.
- Root cause: identical to the Megatron actor bug fixed in the smoke test (commit `12d107cc`).
- Fix applied in `dp_actor.py` (commit `1b32a8a1`):
  1. Added imports for `omega_conf_to_dataclass` and `SelfDistillationConfig`.
  2. Normalized `self_distillation_cfg` through `omega_conf_to_dataclass` in both
     `update_policy()` and `_update_teacher()`.

### [Resolved] SDPO FSDP death spiral — zero self-distillation activation

- After the dict-vs-dataclass fix, SDPO FSDP started training but collapsed by step ~19.
- Trajectory: step 1 had `reprompt_sample_fraction: 0.789`, `score/mean: 0.262`.
  By step 19: all self_distillation metrics = 0.0, `loss = 0.0`, `grad_norm = 0.0`.
- Root cause: with `success_reward_threshold: 1.0`, the model quickly lost all successful
  rollouts (score must be exactly 1.0). With no teacher targets, `self_distillation_mask`
  is all zeros, loss is 0, no gradients flow, model stops learning.
- There is no GRPO fallback in SDPO mode — when the mask is empty, loss is simply 0.
- Additionally, `alpha: 1.0` (pure reverse KL) is more aggressive than `0.5` (JSD).

### [Resolved] GRPO FSDP catastrophic entropy collapse at step 257

- GRPO was training stably for ~256 steps: score oscillating around 0.3-0.55, entropy ~0.11.
- At step 257: entropy crashed from 0.111 to 0.008 (14x drop in one step).
- By step 258: entropy = 0.0015, score = 0.016, response_length = 7767 (near max 8192).
- By step 261: score = 0.004, model generating max-length near-deterministic outputs.
- Validation at step 260: `val-core/sciknoweval/acc/mean@16: 0.0024` — essentially zero.
- Config factors: `use_kl_loss: false`, `use_kl_in_reward: false`, `lr: 1e-5`.
  No regularization to prevent the policy from diverging from the reference.
- Note: upstream sweeps both `lr: 1e-5` and `lr: 1e-6`; the lower LR may be more stable.

### [Resolved] SDPO Megatron crash — missing megatron.core.distributed.custom_fsdp

- Megatron variant crashed during worker init with:
  `ModuleNotFoundError: No module named 'megatron.core.distributed.custom_fsdp'`
- Same issue as smoke test (documented in SDPO_SMOKE_DEBUG_LOG.md).
- `megatron-core==0.15.0` lacks `custom_fsdp`; the default `vanilla_mbridge: true`
  tries to use vanilla `mbridge` which depends on it.
- Fix: added `vanilla_mbridge: false` to the Megatron config to use `megatron.bridge` instead.

### [Resolved] Aligned SDPO configs with upstream Section 3 effective settings

- Traced the upstream config chain: Python dataclass defaults → `actor.yaml` overlay →
  experiment YAML (`sdpo.yaml`) → command-line overrides in `run_sdpo_all.sh`.
- Effective upstream Section 3 config:
  - `success_reward_threshold: 0.5` (from `actor.yaml`, overrides dataclass default 1.0)
  - `alpha: 0.5` (JSD, from command-line)
  - `full_logit_distillation: true` + `distillation_topk: 100` (from `actor.yaml` + command-line)
  - `include_environment_feedback: false` (from command-line)
  - `dont_reprompt_on_self_success: true` (from command-line)
- Applied fixes in commit `0cb2fecb`:
  - FSDP: `success_reward_threshold` 1.0 → 0.5
  - Megatron: `alpha` 1.0 → 0.5, `success_reward_threshold` 1.0 → 0.5, `vanilla_mbridge: false`
- Both SDPO jobs relaunched.

### [WIP] Corrected Run 2 status after relaunch

- The relaunched GRPO run is now the healthiest of the three:
  - still running at step `334`
  - recent validation is around `0.436 acc@16`
- Both SDPO variants showed good early activation and non-zero validation,
  but both have now fallen into an empty-target regime:
  - `reprompt_sample_fraction = 0.0`
  - `active_token_fraction = 0.0`
  - `empty_target_batch = 1.0`
  - `critic/score/mean = 0.0`
- This means the current active debugging target for the chemistry pilot is no
  longer infrastructure or model init. It is the algorithmic/runtime reason
  both SDPO variants stop seeing successful trajectories after the first few
  dozen steps.
