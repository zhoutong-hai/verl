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

## Current Status Snapshot (2026-03-16 23:42 PDT)

- Physics-specific OLMo FSDP configs, run script, and SkyPilot launcher have been added locally.
- The physics launch path has been pushed to your fork branch:
  - commit `6d54110b`
- Future Physics reruns will persist validation samples explicitly:
  - W&B validation table via `trainer.log_val_generations = 8`
  - dumped samples under `$VALIDATION_DATA_DIR`
- The public SciKnowEval Physics split is smaller than Chemistry:
  - `train.json`: `720`
  - `test.json`: `80`
- Physics data is staged and verified on the cluster-accessible shared path:
  - `/hai/zhoutong/section3_physics_assets/data/sciknoweval_physics/train.json`
  - `/hai/zhoutong/section3_physics_assets/data/sciknoweval_physics/test.json`
- The shared OLMo checkpoint path is also verified:
  - `/hai/zhoutong/section3_chemistry_assets/models/Olmo-3-7B-Instruct`
- The physics cluster is up and currently running the patched OLMo SDPO FSDP job on the same cluster:
  - cluster: `verl-olmo3-physics`
  - variant: `sdpo_fsdp`
  - cluster state: `UP`
  - canceled prior jobs on this cluster: `1`, `2`, `3`
  - current job id: `4`
  - node repo commit: `c33e997d`
  - latest observed training step: `12`
  - latest observed validation:
    - step `5`: `val-core/sciknoweval/acc/mean@16 = 0.05`
    - step `5`: `val-core/sciknoweval/acc/best@16/mean = 0.3613375`
    - step `5`: `val-core/sciknoweval/acc/maj@16/mean = 0.0769625`
    - step `10`: `val-core/sciknoweval/acc/mean@16 = 0.03125`
    - step `10`: `val-core/sciknoweval/acc/best@16/mean = 0.2602`
    - step `10`: `val-core/sciknoweval/acc/maj@16/mean = 0.0464`
  - latest observed post-validation train state:
    - step `11`: `self_distillation/reprompt_sample_fraction = 0.13671875`
    - step `11`: `self_distillation/active_token_fraction = 0.13671875`
    - step `11`: `self_distillation/empty_target_batch = 0.86328125`
    - step `11`: `critic/score/mean = 0.01953125`
    - step `11`: `response_length/mean = 2899.3203125`
    - step `12`: `self_distillation/reprompt_sample_fraction = 0.0`
    - step `12`: `self_distillation/active_token_fraction = 0.0`
    - step `12`: `self_distillation/empty_target_batch = 1.0`
    - step `12`: `critic/score/mean = 0.0`
    - step `12`: `response_length/mean = 3240.3828125`
    - step `12`: `actor/grad_norm = 0.0`
- The node-local training log is available at:
  - `/root/sky_logs/4--/tasks/run.log`
- W&B upload is available for the current SDPO relaunch:
  - run id: `31mhmwla`
  - node-local logs remain the most reliable source of truth
- Validation sample dumping is confirmed working on the current SDPO relaunch:
  - log evidence:
    `Dumped generations to /hai/zhoutong/section3_physics_assets/validation_generations/olmo3_7b_section3_physics_sdpo_fsdp_20260316_223616/5.jsonl`
  - dumped file exists:
    `/hai/zhoutong/section3_physics_assets/validation_generations/olmo3_7b_section3_physics_sdpo_fsdp_20260316_223616/5.jsonl`
- The new inline preview path is also confirmed working:
  - `run.log` contains:
    - `[val_sample 1/2 step=5]`
    - `[prompt] ...`
    - `[response] ...`
    - `[ground_truth] B`
    - `[score] 0.0`
    - `[val_sample 2/2 step=5]`
- The capped dump size is confirmed:
  - `5.jsonl` contains exactly `32` rows
  - this is down from the earlier uncapped `1280`-row behavior
- The saved samples are still low quality:
  - the file is present and populated
  - the first inline previews and the first rows of `5.jsonl` still show noisy / derailed generations
  - the run has now collapsed into the zero-target regime by step `12`:
    - `reprompt_sample_fraction = 0.0`
    - `active_token_fraction = 0.0`
    - `empty_target_batch = 1.0`
    - `actor/pg_loss = 0.0`
    - `actor/grad_norm = 0.0`
  - some entries do at least preserve the final `<answer>` tag structure
- The current Physics run is not yet a faithful upstream reproduction for two concrete reasons:
  - runtime config resolved `reward_model.use_reward_loop = True`, while upstream `SDPO` sets `use_reward_loop = False` and explicitly notes that the experimental reward loop gives lower scores
  - the rollout runtime is using `vllm 0.10.0`, and the node log reports:
    `Olmo3ForCausalLM has no vLLM implementation, falling back to Transformers implementation`
  - upstream `SDPO` pins a GH200 stack with `transformers==4.57.1` and `vllm 0.12.0+` for this generation of experiments
  - the fallback diagnosis is now cross-validated:
    - installed node runtime contains `vllm/model_executor/models/qwen3.py`
    - installed node runtime contains `Qwen3ForCausalLM` in the vLLM registry
    - installed node runtime does not contain `olmo3.py` or `Olmo3ForCausalLM`
    - official vLLM `v0.11.0` release notes list `OLMo3` as newly supported, which is consistent with the node still being on `vllm 0.10.0`
- The strongest diagnostic signal is that the run is qualitatively wrong before late-training collapse:
  - step-`5` saved validation samples already contain gibberish / off-task generations
  - this does not look like a subtle SDPO-only regression; it looks like a reproduction/runtime mismatch

## Debug Notes

### [Resolved] Add explicit validation-sample persistence to the Physics setup

- The current OLMo Chemistry runs do not expose prompt/response samples through the accessible logs.
- To avoid depending on reward-manager terminal prints, the Physics configs now persist validation samples directly:
  - `trainer.log_val_generations: 8`
  - `trainer.validation_data_dir: ${oc.env:VALIDATION_DATA_DIR,null}`

### [Resolved] Validation dump path bug identified

- The `main_ppo` launcher process has `VALIDATION_DATA_DIR` set correctly.
- But the Ray `TaskRunner` process that executes `_validate()` does not inherit that env var.
- Because the trainer config used:
  - `trainer.validation_data_dir: ${oc.env:VALIDATION_DATA_DIR,null}`
  the interpolation likely resolves to `null` inside the worker, so `_dump_generations()` is skipped.
- Evidence:
  - validation is definitely running in `/root/sky_logs/1--/tasks/run.log`
  - no `Dumped generations to ...` lines appear
  - no `*.jsonl` files appear anywhere under `/hai/zhoutong/section3_physics_assets`
  - the top-level `python3 -m verl.trainer.main_ppo` process has `VALIDATION_DATA_DIR`
  - the Ray `TaskRunner` process does not
- Fix applied locally:
  - the OLMo run scripts now pass a literal CLI override:
    - `trainer.validation_data_dir="$VALIDATION_DATA_DIR"`
  - patched files:
    - `examples/sdpo_trainer/run_olmo3_7b_sciknoweval_physics_section3.sh`
    - `examples/sdpo_trainer/run_olmo3_7b_sciknoweval_chemistry_section3.sh`
- Relaunch confirmation:
  - the patched Physics GRPO run was relaunched as job `2`
  - `/root/sky_logs/2--/tasks/run.log` now shows:
    `Dumped generations to /hai/zhoutong/section3_physics_assets/validation_generations/olmo3_7b_section3_physics_grpo_fsdp_20260316_220034/5.jsonl`
  - the dumped file exists and is non-empty on disk

### [Resolved] Add capped validation dumps and inline sample previews

- A second relaunch was needed because the first SDPO attempt on job `3` started before the inline-preview patch had been pushed.
- The patched code was committed and pushed in:
  - `c33e997d` `Limit validation sample dumps and print inline previews`
- The current SDPO relaunch on job `4` confirms all intended behaviors:
  - resolved config contains:
    - `print_val_generations: 2`
    - `validation_console_max_chars: 1200`
    - `validation_dump_generations: 32`
  - `run.log` now prints inline validation previews
  - `5.jsonl` is capped to `32` rows instead of dumping the full repeated validation set

### [WIP] Review saved validation sample quality

- The dump mechanism is working, but the first saved samples at step `5` are not yet healthy.
- The first few entries in `5.jsonl` show:
  - correct prompt preservation
  - model outputs that derail into unrelated physics text and noisy token sequences
  - `score = 0.0` and `acc = 0.0`
- This is useful because we now have concrete artifacts to inspect, but it also suggests the early-run model behavior is still poor even though the logging plumbing is fixed.

### [WIP] Resolve paper-reproduction mismatches before trusting Physics SDPO results

- Two concrete mismatches with the upstream `SDPO` setup are now confirmed:
  - our live Physics run resolves `reward_model.use_reward_loop = True`
  - upstream `SDPO/verl/trainer/config/user.yaml` sets `use_reward_loop: False` with the note:
    `disables experimental reward manager (which gives lower scores)`
- The rollout runtime also differs materially from the upstream GH200 stack:
  - current SkyPilot image:
    `verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2`
  - setup upgrades `transformers` to `4.57.1`, but does not upgrade `vllm`
  - node runtime therefore stays on `vllm 0.10.0`
  - node log confirms OLMo3 is not properly supported there and falls back away from native vLLM
- These mismatches are now the leading explanation for why the current Physics OLMo run differs so sharply from the paper-style behavior.

### [Resolved] Cross-validate OLMo3 vs Qwen3 support in the live vLLM runtime

- The installed node runtime is `vllm 0.10.0`.
- Direct inspection of the installed package on the node shows:
  - `vllm/model_executor/models/qwen3.py` exists
  - `Qwen3ForCausalLM` is present in the local vLLM registry
  - only `olmo.py`, `olmo2.py`, and `olmoe.py` exist; there is no `olmo3.py`
- This matches the node warning in `run.log`:
  - `Olmo3ForCausalLM has no vLLM implementation, falling back to Transformers implementation`
- Conclusion:
  - the OLMo mismatch is real and specific to the current runtime
  - `Qwen3` should be a cleaner next model to test on the same cluster/image because it is natively recognized by the installed vLLM package

### [WIP] Stage Physics data under `/hai/zhoutong`

- The launcher expects:
  - `/hai/zhoutong/section3_physics_assets/data/sciknoweval_physics/train.json`
  - `/hai/zhoutong/section3_physics_assets/data/sciknoweval_physics/test.json`
- The public `physics` split has now been copied from the local `SDPO` repo onto the live cluster and verified with `wc -l`.
- The launch now reuses the existing shared OLMo checkpoint path under `/hai/zhoutong`.

### [Resolved] GRPO baseline launch reached active training

- `sky launch -c verl-olmo3-physics ... --env VARIANT=grpo_fsdp` succeeded.
- The cluster is `UP`, job `1` is running, and the run has passed setup, Ray startup, vLLM startup, and validation.
- Current operational caveat:
  - vLLM falls back to the Transformers implementation for OLMo
  - W&B upload is flaky, so `/root/sky_logs/1--/tasks/run.log` is the primary log source
