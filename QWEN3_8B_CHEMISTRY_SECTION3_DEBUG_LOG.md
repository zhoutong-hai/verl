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

## Current Status Snapshot (2026-03-16, refreshed)

### Active Runs

The authoritative live state is:

| Cluster | Variant | Job | Status | Notes |
|---------|---------|-----|--------|-------|
| `verl-qwen3-chemistry-grpo` | GRPO FSDP | `5` | `RUNNING` | Fairer rerun with upstream-aligned FSDP config |
| `verl-qwen3-chemistry-sdpo` | SDPO FSDP | `4` | `RUNNING` | Fairer rerun with upstream-aligned FSDP config |
| `verl-qwen3-chemistry-sdpo-megatron` | SDPO Megatron | `3` | `RUNNING` | Historical/debug arm still active on the older Megatron config |

### Latest Live Metrics

#### GRPO FSDP (`verl-qwen3-chemistry-grpo`, job `5`)

- Current visible step: `108`
- Best visible validation so far: at step `105`
  - `val-core/sciknoweval/acc/mean@16 = 0.4643`
  - `val-core/sciknoweval/acc/best@16/mean = 0.7389`
  - `val-core/sciknoweval/acc/maj@16/mean = 0.5238`
- Latest training health:
  - `critic/score/mean = 0.5508`
  - `actor/entropy = 0.0210`
  - `perf/throughput = 1300.9`
- Main concern:
  - `response_length/mean = 8192.0`
  - `response_length/clip_ratio = 1.0`
  - GRPO is learning, but it is currently maxing out the response budget on every visible batch.

#### SDPO FSDP (`verl-qwen3-chemistry-sdpo`, job `4`)

- Current visible step: `74`
- Latest visible validation: at step `30`
  - `val-core/sciknoweval/acc/mean@16 = 0.0`
- Latest training health:
  - `self_distillation/success_group_fraction = 0.0`
  - `self_distillation/reprompt_sample_fraction = 0.0`
  - `self_distillation/active_token_fraction = 0.0`
  - `self_distillation/empty_target_batch = 1.0`
  - `actor/pg_loss = 0.0`
  - `actor/grad_norm = 0.0`
- Main concern:
  - the run is operationally alive, but it is still in the zero-target / zero-learning regime.

Validated score trajectory from W&B run `mw7q3350`:

| Step | `acc@16` | `success_group_fraction` | `reprompt_sample_fraction` | `critic/score/mean` | `response_length/mean` |
|------|----------|--------------------------|----------------------------|---------------------|------------------------|
| `5` | `0.2527` | `0.8125` | `0.7656` | `0.2031` | `1066.9` |
| `10` | `0.0179` | `0.0625` | `0.0547` | `0.0078` | `1972.9` |
| `15` | `0.0030` | `0.0` | `0.0` | `0.0` | `2983.4` |
| `20` | `0.0003` | `0.0` | `0.0` | `0.0` | `3255.2` |
| `25` | `0.0` | `0.0` | `0.0` | `0.0` | `3970.8` |
| `30+` | `0.0` | `0.0` | `0.0` | `0.0` | `4043+` |

#### SDPO Megatron (`verl-qwen3-chemistry-sdpo-megatron`, job `3`)

- Current visible step: `119`
- Latest visible training health:
  - `self_distillation/success_group_fraction = 0.0`
  - `self_distillation/reprompt_sample_fraction = 0.0`
  - `self_distillation/active_token_fraction = 0.0`
  - `self_distillation/empty_target_batch = 1.0`
  - `actor/pg_loss = 0.0`
  - `actor/grad_norm = 0.0`
  - `self_distillation/teacher_update_rate = 0.05`
  - teacher/actor parameter drift metrics are still being logged, so EMA is active
- Main concern:
  - like FSDP SDPO, Megatron SDPO is still running but has no active SDPO supervision in the visible window.

Validated score trajectory from W&B run `i7ydwmwo`:

| Step | `acc@16` | `success_group_fraction` | `reprompt_sample_fraction` | `critic/score/mean` | `response_length/mean` |
|------|----------|--------------------------|----------------------------|---------------------|------------------------|
| `5` | `0.2735` | `0.8125` | `0.7656` | `0.2070` | `773.5` |
| `10` | `0.0223` | `0.3750` | `0.3398` | `0.0625` | `2050.2` |
| `15` | `0.0015` | `0.1250` | `0.1133` | `0.0195` | `6051.8` |
| `20` | `0.0` | `0.0` | `0.0` | `0.0` | `5641.0` |
| `25+` | `0.0` | `0.0` | `0.0` | `0.0` | `5460+` |

### Current Diagnosis Of The SDPO Collapse

The collapse pattern is now directly verified for both SDPO runs:

1. The runs start with a healthy pocket of successful samples.
2. Within 5-10 validation intervals, both the reward signal and the SDPO target supply collapse.
3. Once `success_group_fraction` hits `0`, `reprompt_sample_fraction` also hits `0`.
4. That makes `self_distillation_mask` empty, so SDPO loss becomes exactly `0`.
5. After that, `actor/pg_loss`, `actor/grad_norm`, and `critic/score/mean` all stay at `0`, so the run cannot recover on its own.

The most concrete correlated symptoms are:

- response length rises sharply as accuracy collapses
- the Chemistry XML answer-format metric also collapses to nearly `0` early

For the current FSDP run, the validation-side format metric goes:

- step `5`: `0.4060`
- step `10`: `0.0012`
- step `15+`: `0.0`

For the current Megatron run, it goes:

- step `5`: `0.5688`
- step `10`: `0.0176`
- step `15`: `0.0003`
- step `20+`: `0.0`

Important note: in [mcq.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/mcq.py), `incorrect_format` is currently misnamed. A logged value of `1` actually means the response matched the required trailing XML answer format.

### Historical Summary

#### Run 1: initial chemistry pilot

All three first-wave runs failed or collapsed:

| Cluster | Variant | Outcome |
|---------|---------|---------|
| `verl-qwen3-chemistry-grpo` | GRPO FSDP | Entropy collapse at step `257` |
| `verl-qwen3-chemistry-sdpo` | SDPO FSDP | Zero-target death spiral by step `~19` |
| `verl-qwen3-chemistry-sdpo-megatron` | SDPO Megatron | Init crash on `megatron.core.distributed.custom_fsdp` |

#### Run 2: corrected SDPO config, before final FSDP fairness cleanup

- GRPO became much healthier and reached roughly `0.48 acc@16` by step `340+`.
- Both SDPO variants still collapsed after showing good early activation.
- Megatron remained useful for debugging, but the fairest paper-style comparison should focus on FSDP.

#### Run 3: current fairer FSDP reruns

These reruns incorporate the final FSDP alignment changes:

- `max_model_len=18944` for GRPO
- `clip_ratio_high=0.28` for both FSDP configs
- `remove_thinking_from_demonstration=true` for FSDP SDPO
- paper-style evaluation target: best `val-core/sciknoweval/acc/mean@16` within `1h` and `5h`

### Current FSDP Config Summary

#### GRPO FSDP

| Parameter | Value |
|-----------|-------|
| `train_batch_size` | `32` |
| `rollout.n` | `8` |
| `ppo_mini_batch_size` | `32` |
| `lr` | `1e-5` |
| `clip_ratio_high` | `0.28` |
| `max_model_len` | `18944` |
| `use_kl_loss` | `false` |
| `use_kl_in_reward` | `false` |

#### SDPO FSDP

| Parameter | Value |
|-----------|-------|
| `train_batch_size` | `32` |
| `rollout.n` | `8` |
| `ppo_mini_batch_size` | `32` |
| `lr` | `1e-5` |
| `clip_ratio_high` | `0.28` |
| `alpha` | `0.5` |
| `full_logit_distillation` | `true` |
| `distillation_topk` | `100` |
| `success_reward_threshold` | `0.5` |
| `teacher_update_rate` | `0.05` |
| `dont_reprompt_on_self_success` | `true` |
| `remove_thinking_from_demonstration` | `true` |
| `include_environment_feedback` | `false` |

### Verified Upstream OLMo Reference

Verified from public W&B run [`jonhue/SDPO/xjiucmxw`](https://wandb.ai/jonhue/SDPO/runs/xjiucmxw):

- model: `allenai/Olmo-3-7B-Instruct`
- dataset: `sciknoweval/chemistry_filtered`
- hardware: `1` node, `4` GPUs
- run state: `crashed`
- matching SDPO knobs:
  - `alpha=0.5`
  - `full_logit_distillation=true`
  - `distillation_topk=100`
  - `success_reward_threshold=0.5`
  - `remove_thinking_from_demonstration=true`
  - `clip_ratio_high=0.28`
  - `loss_agg_mode=token-mean`
  - validation `n=16`

Verified OLMo metric points:

| Metric | Step `5` | Step `40` | Step `50` | Step `150` |
|--------|----------|-----------|-----------|------------|
| `critic/score/mean` | `0.2930` | `0.7148` | `0.6445` | `0.7383` |
| `actor/entropy` | `0.9173` | `1.1950` | `1.0797` | `1.1535` |
| `response_length/mean` | `692.72` | `210.55` | `202.92` | `166.54` |
| `response_length/clip_ratio` | `0.0039` | `0.0` | `0.0` | `0.0` |
| `success_group_fraction` | `0.7188` | `0.7813` | `0.8125` | `0.8125` |
| `reprompt_sample_fraction` | `0.6992` | `0.7773` | `0.8008` | `0.8047` |
| `val acc@16` | `0.3592` | `0.6967` | `0.7116` | `0.7961` |

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
  - Megatron: `success_reward_threshold` 1.0 → 0.5, `vanilla_mbridge: false`
  - Note: the current Megatron chemistry config still uses `alpha: 1.0`
    because `full_logit_distillation: false` remains a Megatron limitation.
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

### [WIP] FSDP-only reruns launched with fairer upstream alignment

- Pushed config alignment commit: `9a1dbf9c`
- Reused the same cluster names to preserve history:
  - `verl-qwen3-chemistry-grpo`
  - `verl-qwen3-chemistry-sdpo`
- Canceled the blocking previous jobs before relaunch:
  - GRPO: canceled job `2`, new job is `5`
  - SDPO: canceled job `3`, new job is `4`
- Current relaunch snapshot:
  - `verl-qwen3-chemistry-grpo` job `5`: `RUNNING`
  - `verl-qwen3-chemistry-sdpo` job `4`: `RUNNING`
- These reruns include the final FSDP alignment changes:
  - `remove_thinking_from_demonstration=true` for SDPO
  - `clip_ratio_high=0.28` for both FSDP configs
  - `max_model_len=18944` for GRPO
- Evaluation target for this round:
  - compare the best `val-core/sciknoweval/acc/mean@16` within the first `1h`
  - compare again within the first `5h`

### [WIP] Cross-validated hypothesis: long-response dilution is a major contributor to SDPO collapse

#### Summary

Both SDPO variants (FSDP and Megatron) collapse to zero score within ~12 training
steps despite starting with healthy self-distillation activation. A strong
working hypothesis is that Qwen3-8B's much longer responses make the SDPO loss
harder to optimize: our Qwen3 run often produces multi-thousand-token responses,
while the public author run on `allenai/Olmo-3-7B-Instruct` converges to short
~200-token responses. Because SDPO uses token-mean aggregation over all active
response tokens, long reasoning traces can dilute answer-relevant supervision.

This hypothesis is supported by code inspection and by the public OLMo run, but
it is **not yet proven as the unique root cause**. The comparison is confounded
by both model choice (`Qwen3-8B` vs `Olmo-3-7B-Instruct`) and dataset split
(`chemistry` vs `chemistry_filtered`).

#### SDPO FSDP (Qwen3-8B) — Step-by-step collapse trajectory

The SDPO FSDP run (job 3, corrected config with `success_reward_threshold: 0.5`)
**did NOT cold-start**. It began with strong self-distillation activation that
then rapidly collapsed:

| Step | Score | Success Groups | Entropy | pg_loss | grad_norm | Active Token Frac |
|------|-------|---------------|---------|---------|-----------|-------------------|
| 1    | 0.227 | **71.9%** | 1.53 | 0.436 | 1.03 | 69.5% |
| 2    | 0.246 | **71.9%** | 1.71 | 0.256 | 0.58 | 69.5% |
| 3    | 0.188 | **84.4%** | 1.87 | 0.444 | 1.20 | 80.1% |
| 4    | 0.231 | **93.8%** | 1.85 | 0.425 | 0.81 | 87.9% |
| 5    | 0.180 | **78.1%** | 1.93 | 0.326 | 0.70 | 73.4% |
| 6    | 0.148 | **71.9%** | 1.54 | 0.259 | 0.65 | 67.6% |
| 7    | 0.109 | **62.5%** | 1.69 | 0.327 | 0.62 | 57.0% |
| 8    | 0.051 | 21.9% | 2.23 | 0.015 | 0.08 | 21.1% |
| 9    | 0.008 | 6.3% | 2.33 | 0.002 | 0.02 | 5.5% |
| 10   | 0.012 | 9.4% | 2.76 | 0.004 | 0.05 | 8.2% |
| 11   | 0.004 | 3.1% | 2.91 | 0.000 | 0.01 | 2.7% |
| 12   | 0.008 | 6.3% | 3.04 | 0.004 | 0.04 | 5.5% |
| **13** | **0.0** | **0%** | 3.11 | 0.0 | 0.0 | 0% |
| 14+  | **0.0** | **0%** | ~4.5+ | 0.0 | 0.0 | 0% |

The **entropy explosion** from 1.53 → 4.6+ is the smoking gun: the distillation
loss is making the model *more random*, not more focused. Once the model
becomes too random to score on any rollout, `self_distillation_mask` goes to
all zeros and the model is permanently stuck (no GRPO fallback in SDPO mode).

#### Verified public OLMo-3-7B-Instruct SDPO reference run

Reference: W&B run `jonhue/SDPO/xjiucmxw`
(`FINAL-SDPO-train32-alpha0.5-rollout8-lr1e-5-drossTrue-allenai-Olmo-3-7B-Instruct`)

The following items are directly verified from the public W&B API:

- run state: `crashed`
- model: `allenai/Olmo-3-7B-Instruct`
- dataset: `sciknoweval/chemistry_filtered`
- hardware: `1` node, `4` GPUs
- SDPO config:
  - `alpha: 0.5`
  - `full_logit_distillation: true`
  - `distillation_topk: 100`
  - `success_reward_threshold: 0.5`
  - `dont_reprompt_on_self_success: true`
  - `remove_thinking_from_demonstration: true`
  - `clip_ratio_high: 0.28`
  - `loss_agg_mode: token-mean`
  - validation `n: 16`

Verified OLMo metric points from the public run:

| Metric | OLMo step 5 | OLMo step 40 | OLMo step 50 | OLMo step 150 |
|--------|-------------|--------------|--------------|---------------|
| `critic/score/mean` | 0.2930 | 0.7148 | 0.6445 | 0.7383 |
| `actor/entropy` | 0.9173 | 1.1950 | 1.0797 | 1.1535 |
| `response_length/mean` | 692.72 | 210.55 | 202.92 | 166.54 |
| `response_length/clip_ratio` | 0.0039 | 0.0 | 0.0 | 0.0 |
| `success_group_fraction` | 0.7188 | 0.7813 | 0.8125 | 0.8125 |
| `reprompt_sample_fraction` | 0.6992 | 0.7773 | 0.8008 | 0.8047 |
| `actor/pg_loss` | 0.0529 | 0.0220 | 0.0196 | 0.0098 |
| `actor/grad_norm` | 0.2187 | 0.1022 | 0.0893 | 0.0626 |
| `val acc@16` | 0.3592 | 0.6967 | 0.7116 | 0.7961 |
| `val best@16` | 0.7748 | 0.7814 | 0.7867 | 0.8334 |

By contrast, our Qwen3 SDPO run collapses from healthy early activation to zero
score by roughly step `13`, while still producing far longer responses. The
response-length gap is real and large, but it should be interpreted as a strong
correlate, not yet a proven single-cause explanation.

#### Root cause analysis

Five compounding factors were identified by tracing the full distillation loss
computation path:

**1. Token-mean aggregation dilutes the answer signal (~34x)**

The loss uses `loss_agg_mode: "token-mean"` (the default, not overridden in any
chemistry config):

```
loss = sum(per_token_loss for all active tokens) / total_active_tokens
```

All response tokens contribute equally. With Qwen3-8B's ~5800 tokens per response
(vs OLMo's ~170), the gradient from answer-relevant tokens is diluted by a factor
of ~34x. There is no mechanism to weight answer tokens differently from
thinking/filler tokens.

Source: `verl/trainer/ppo/core_algos.py` lines 965–970, `agg_loss()` at lines
773–828.

**2. Self-distillation mask is per-sample, not per-token**

The `self_distillation_mask` is a `[batch_size]` tensor that masks entire
sequences. When a sample is selected for distillation, ALL of its ~5800
response tokens contribute to the loss.

Source: `verl/trainer/ppo/ray_trainer.py` lines 798–802 (mask construction),
`core_algos.py` lines 848–849 (`loss_mask * self_distillation_mask.unsqueeze(1)`).

**3. Teacher sees an enriched prompt (context mismatch)**

The teacher evaluates the student's response tokens under a *different* prompt
that includes a successful sibling's full response as a demonstration:

```
{original prompt}

Correct solution:

{successful sibling's full response (~5800 tokens)}

Correctly solve the original question.
```

The student generated its tokens under the original prompt only. This context
mismatch means the teacher and student assign very different probabilities to
thinking/reasoning tokens. The distillation loss pushes the student to match
the teacher's context-mismatched distribution across all 5800 tokens.

Source: `verl/trainer/ppo/ray_trainer.py` lines 728–756
(`_build_teacher_message`), lines 780–784 (teacher input construction).

**4. JSD over top-100 logits on long noisy responses may encourage entropy growth**

With `full_logit_distillation: true`, `alpha: 0.5` (JSD), and
`distillation_topk: 100`, the per-token Jensen–Shannon divergence is computed
over the top-100 vocabulary items. On tokens where teacher and student contexts
differ sharply, this may push the student toward a more diffuse distribution.
This is a plausible explanation for the observed entropy growth, but we do not
yet have a controlled ablation proving that this is the dominant driver.

Source: `verl/trainer/ppo/core_algos.py` lines 857–906 (top-k JSD
computation).

**5. SDPO completely bypasses the GRPO policy gradient**

When `loss_mode="sdpo"`, the actor enters the `if self_distillation_enabled:`
branch and **completely skips** the standard GRPO policy loss. The only
learning signal comes from `compute_self_distillation_loss()`. Once the
distillation mask goes to all zeros (no successful rollouts), the model
receives exactly zero gradient — there is no GRPO fallback.

Source: `verl/workers/actor/dp_actor.py` lines 819–860 (SDPO branch) vs
lines 861–876 (GRPO `else` branch).

#### Additional notes on FSDP vs Megatron teacher paths

- **FSDP** (`teacher_scoring_mode: actor_worker`): Uses a true EMA teacher
  (`teacher_update_rate: 0.05`) that tracks the student. As the student
  degrades, the teacher follows within ~20 steps → death spiral.
  Source: `verl/workers/actor/dp_actor.py` lines 133–156 (`_update_teacher`).

- **Megatron** (`teacher_scoring_mode: trainer_ref`): Uses the trainer-side
  reference path for teacher scoring, but the underlying `ref_module` on the
  Megatron worker is still updated by EMA in `megatron_workers.py`. So this is
  **not a frozen teacher** in the current branch.

#### Metric definitions

For reference, the key self-distillation metrics are computed as follows
(source: `verl/trainer/ppo/ray_trainer.py` lines 811–822):

- **`success_group_fraction`**: Fraction of unique UIDs (prompts) that have at
  least one rollout scoring ≥ `success_reward_threshold`. A "group" is all
  `n=8` rollouts from the same prompt.

- **`success_sample_fraction`**: Fraction of individual samples that were
  assigned a distillation target (a successful sibling's response). With
  `dont_reprompt_on_self_success: true`, a sample's own success doesn't count —
  it needs a *different* rollout in its group to have succeeded.

- **`active_token_fraction`**: Fraction of all response tokens that are inside
  samples with `self_distillation_mask = 1` (i.e., samples eligible for
  distillation).

- **`empty_target_batch`**: `1.0` when `self_distillation_mask.sum() == 0`
  (no samples in the batch have a distillation target). When this is `1.0`,
  the loss is exactly 0.0 and no gradient flows.

Source for `_collect_solutions_by_uid`: `ray_trainer.py` lines 620–629.
The threshold comparison is on `reward_tensor.sum(dim=-1)` (per-sequence
reward sum).

### [Resolved] Cross-validation review of the OLMo-based diagnosis

- The following parts are well supported by the code:
  - `loss_agg_mode: token-mean` does average over all active response tokens.
  - `self_distillation_mask` is per-sample and is broadcast across tokens.
  - the teacher scores the student's response under an enriched reprompted context.
  - SDPO bypasses the normal GRPO policy-gradient branch when `loss_mode="sdpo"`.
- The following parts should be treated as hypotheses, not established root causes:
  - "long-response dilution is the root cause" is plausible, but not isolated.
    The comparison is confounded by both model choice (`Qwen3-8B` vs
    `Olmo-3-7B-Instruct`) and dataset split (`chemistry` vs the earlier observed
    `chemistry_filtered` upstream path).
  - the entropy-explosion explanation for top-k JSD on long responses is
    plausible, but we do not yet have a controlled ablation proving it.
- One claim in the finding is incorrect for this branch:
  - Megatron `teacher_scoring_mode: trainer_ref` does **not** imply a frozen
    teacher. The Megatron worker still applies EMA updates to `ref_module` in
    `megatron_workers.py`, so the teacher can drift and improve over time.
- There are also stale/inconsistent entries elsewhere in this log that should
  not be treated as the latest truth:
  - the earlier "Live cluster state" section still references pre-rerun job IDs
    `2/3`, while the current reruns are job `5` (GRPO) and job `4` (SDPO).
  - the earlier "Fixes applied" note says Megatron `alpha` was moved to `0.5`,
    but the current Megatron chemistry config is `alpha: 1.0` because
    `full_logit_distillation: false` is still enforced there.
- Extra caution on metric interpretation:
  - the `incorrect_format` field in `feedback/mcq.py` is currently named
    misleadingly; it is set to `1` when the answer format is actually correct.
