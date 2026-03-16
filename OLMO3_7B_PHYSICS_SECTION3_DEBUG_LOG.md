# OLMo-3-7B Physics Section 3 Debug Log

## Goal

Reproduce the paper-style Physics Section 3 setup with `allenai/Olmo-3-7B-Instruct` on the current branch, focusing on the fair FSDP comparison:

1. on-policy GRPO baseline
2. upstream-style SDPO FSDP

This run is intended as an easier Section 3 debugging task than Chemistry while keeping the same no-rich-feedback MCQ setup.

## Original Task

Switch the OLMo Section 3 experiment to the public SciKnowEval `physics` split and make sure validation sample outputs are explicitly persisted for review.

## Working Plan

1. Stage the Physics dataset into `/hai/zhoutong/section3_physics_assets/data/sciknoweval_physics`.
2. Add OLMo-specific FSDP Physics configs and a dedicated SkyPilot launcher.
3. Launch GRPO and SDPO FSDP runs on Physics, one at a time on the available node.
4. Track the same `1h` / `5h` validation windows used in the paper-style comparison.
5. Persist validation samples via:
   - `trainer.log_val_generations = 8`
   - `trainer.validation_data_dir = $VALIDATION_DATA_DIR`

## Commands

### SkyPilot: OLMo Physics GRPO FSDP

```bash
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
export WANDB_API_KEY='***'
sky launch -c verl-olmo3-physics \
  /Users/zhoutong/code/verl/examples/skypilot/verl-olmo3-section3-physics.yaml \
  --env VARIANT=grpo_fsdp \
  --secret WANDB_API_KEY -y
```

### SkyPilot: OLMo Physics SDPO FSDP

```bash
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
export WANDB_API_KEY='***'
sky launch -c verl-olmo3-physics \
  /Users/zhoutong/code/verl/examples/skypilot/verl-olmo3-section3-physics.yaml \
  --env VARIANT=sdpo_fsdp \
  --secret WANDB_API_KEY -y
```

## Monitoring Metrics

- `val-core/sciknoweval/acc/mean@16`
- `val-core/sciknoweval/acc/best@16/mean`
- `val-core/sciknoweval/acc/maj@16/mean`
- `self_distillation/reprompt_sample_fraction`
- `self_distillation/active_token_fraction`
- `self_distillation/empty_target_batch`
- `critic/score/mean`
- `response_length/mean`
- `response_length/clip_ratio`
- `actor/entropy`
- `perf/throughput`

## Current Status Snapshot (2026-03-16 17:05 PDT)

- Physics-specific OLMo FSDP configs, run script, and SkyPilot launcher have been added locally.
- Future Physics reruns will persist validation samples explicitly:
  - W&B validation table via `trainer.log_val_generations = 8`
  - dumped samples under `$VALIDATION_DATA_DIR`
- The public SciKnowEval Physics split is smaller than Chemistry:
  - `train.json`: `720`
  - `test.json`: `80`
- Physics data still needs to be staged into `/hai/zhoutong/section3_physics_assets/data/sciknoweval_physics` before launch.
- No Physics training job has been launched yet.

## Debug Notes

### [Resolved] Add explicit validation-sample persistence to the Physics setup

- The current OLMo Chemistry runs do not expose prompt/response samples through the accessible logs.
- To avoid depending on reward-manager terminal prints, the Physics configs now persist validation samples directly:
  - `trainer.log_val_generations: 8`
  - `trainer.validation_data_dir: ${oc.env:VALIDATION_DATA_DIR,null}`

### [WIP] Stage Physics data under `/hai/zhoutong`

- The launcher expects:
  - `/hai/zhoutong/section3_physics_assets/data/sciknoweval_physics/train.json`
  - `/hai/zhoutong/section3_physics_assets/data/sciknoweval_physics/test.json`
- The next step is to copy the public `physics` split from the local `SDPO` repo onto the live cluster and reuse the existing shared OLMo checkpoint path under `/hai/zhoutong`.
