# Qwen3-8B Physics Section 3 Debug Log

## Goal

Run the same public SciKnowEval `physics` Section 3 setup on the public Qwen3 8B checkpoint using the existing Physics launcher path, mainly to rule out the OLMo-specific vLLM fallback mismatch.

## Original Task

After logging the OLMo Physics findings, continue with the same Physics launch path but switch the model away from OLMo and onto the public Qwen3 8B checkpoint.

## Working Plan

1. Preserve the OLMo findings in the OLMo Physics debug log.
2. Reuse the same `verl-olmo3-physics` cluster and launcher path.
3. Override the model settings to:
   - `MODEL_REPO_ID=Qwen/Qwen3-8B`
   - `MODEL_PATH=/hai/zhoutong/section3_physics_assets/models/Qwen3-8B`
   - `RUN_NAME_PREFIX=qwen3_8b_section3_physics`
4. Keep the Physics data, validation dumps, and inline sample printing unchanged.
5. Check whether the early generations are cleaner than the OLMo run on the same runtime.

## Commands

```bash
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
export WANDB_API_KEY='***'
sky cancel verl-olmo3-physics 4
sky launch -c verl-olmo3-physics \
  /Users/zhoutong/code/verl/examples/skypilot/verl-olmo3-section3-physics.yaml \
  --env VARIANT=sdpo_fsdp \
  --env MODEL_REPO_ID=Qwen/Qwen3-8B \
  --env MODEL_PATH=/hai/zhoutong/section3_physics_assets/models/Qwen3-8B \
  --env RUN_NAME_PREFIX=qwen3_8b_section3_physics \
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
- inline validation samples in `run.log`
- capped validation dump files under `/hai/zhoutong/section3_physics_assets/validation_generations/`

## Current Status Snapshot (2026-03-17 07:01 PDT)

- The Qwen Physics SDPO FSDP run succeeded as a strong reproduction sanity check on the same cluster image that failed for OLMo.
- The run was healthy through at least step `306` and was then canceled intentionally to free the cluster for the Megatron comparison:
  - cluster: `verl-olmo3-physics`
  - completed live job: `6`
  - status now: canceled by request, cluster kept `UP`
- The latest strong validation checkpoint before cancellation was step `305`:
  - `val-core/sciknoweval/acc/mean@16 = 0.76171875`
  - `val-core/sciknoweval/acc/best@16/mean = 0.8374625`
  - `val-core/sciknoweval/acc/maj@16/mean = 0.7811375`
- The SDPO supervision supply stayed abundant instead of collapsing:
  - step `305`: `success_group_fraction = 0.9375`, `reprompt_sample_fraction = 0.9375`, `empty_target_batch = 0.0625`
  - step `306`: `success_group_fraction = 1.0`, `reprompt_sample_fraction = 0.99609375`, `empty_target_batch = 0.00390625`
- Generations remained short and sane:
  - step `305`: `response_length/mean = 395.1484375`, `max = 964`, `clip_ratio = 0.0`
  - step `306`: `response_length/mean = 431.9609375`, `max = 1046`, `clip_ratio = 0.0`
- Inline validation samples and capped dump files both worked on this run:
  - `run.log` printed `[val_sample ...]`, `[prompt]`, `[response]`, `[ground_truth]`, `[score]`
  - dumps were written under `/hai/zhoutong/section3_physics_assets/validation_generations/qwen3_8b_section3_physics_sdpo_fsdp_20260317_011603/`
- The strongest current conclusion is:
  - FSDP SDPO on public Qwen3 8B Physics works on this environment
  - the earlier OLMo failure was very likely dominated by the runtime/model-stack mismatch rather than a general SDPO implementation failure
- Next experiment:
  - reuse the same Physics launch path
  - keep the same public Qwen3 8B model, Physics data, inline samples, and capped validation dumps
  - rerun Megatron SDPO on the new full-logit top-k implementation
  - compare that refreshed Megatron result against this healthy FSDP SDPO baseline
- Current relaunch status:
  - the full-logit Megatron Physics rerun is now active on the existing `verl-olmo3-physics` node via a manual remote launch
  - remote repo path: `/root/verl-section3-physics`
  - remote repo commit: `9bec3e12`
  - manual remote log: `/root/sky_logs/manual-megatron-20260317_184636/run.log`
  - manual launcher PID: `1475282`
  - confirmed from the live `run.log` config dump:
    - `teacher_scoring_mode = trainer_ref`
    - `full_logit_distillation = true`
    - `distillation_topk = 100`
    - `alpha = 0.5`
- Megatron comparison status:
  - first attempt: job `7`
  - result: failed immediately because the cluster cloned an older branch state that did not yet include the local `sdpo_megatron` Physics launcher changes
  - fix: committed and pushed the launcher/config support as `81ee904a Add Qwen physics Megatron SDPO launcher`
  - retry: job `8`
  - final status:
    - the run finished the full `660` steps without a traceback
    - best observed validation was early:
      - step `5`
      - `val-core/sciknoweval/acc/mean@16 = 0.5859375`
      - `val-core/sciknoweval/acc/best@16/mean = 0.8080375`
      - `val-core/sciknoweval/acc/maj@16/mean = 0.6187375`
    - late training collapsed to a degenerate zero-reward regime:
      - step `660`
      - `val-core/sciknoweval/acc/mean@16 = 0.0`
      - `success_group_fraction = 0.0`
      - `reprompt_sample_fraction = 0.0`
      - `empty_target_batch = 1.0`
      - `response_length/mean = 1.0`
      - `actor/pg_loss = 0.0`
      - `actor/grad_norm = 0.0`
    - this is much worse than the FSDP SDPO baseline on the same Qwen Physics task

## Debug Notes

### [Resolved] OLMo-specific vLLM mismatch motivates the Qwen switch

- The current node runtime supports `Qwen3ForCausalLM` but not `Olmo3ForCausalLM`.
- That means a Qwen-instruct run on the same cluster is a cleaner runtime sanity check than continuing to debug OLMo on `vllm 0.10.0`.

### [Resolved] `Qwen/Qwen3-8B-Instruct` is not the public repo name

- Reusing the same launcher with:
  - `MODEL_REPO_ID=Qwen/Qwen3-8B-Instruct`
  - `MODEL_PATH=/hai/zhoutong/section3_physics_assets/models/Qwen3-8B-Instruct`
  produced a Hugging Face `Repository Not Found` error during setup.
- Current working assumption:
  - the official public repo is `Qwen/Qwen3-8B`
  - Qwen3 does not use a separate public `-Instruct` repo name in the same way as OLMo

### [WIP] Launch the Qwen3 8B Physics SDPO run

- The plan is to reuse:
  - the same cluster name
  - the same Physics dataset
  - the same SDPO/FSDP config shape
  - the same validation preview and dump plumbing
- Only the model repo, model path, and run prefix are changing for this next run.
- Current live attempt:
  - job `6`
  - `MODEL_REPO_ID=Qwen/Qwen3-8B`
  - `MODEL_PATH=/hai/zhoutong/section3_physics_assets/models/Qwen3-8B`
  - this run ended up as a strong positive result rather than a mere startup sanity check

### [Resolved] Qwen Physics FSDP SDPO is healthy on this runtime

- On the same cluster image where OLMo Physics collapsed, the public Qwen3 8B run stayed healthy through at least `global_step=306`.
- The strongest observed validation point before cancellation was:
  - step `305`
  - `val-core/sciknoweval/acc/mean@16 = 0.76171875`
  - `val-core/sciknoweval/acc/best@16/mean = 0.8374625`
  - `val-core/sciknoweval/acc/maj@16/mean = 0.7811375`
- SDPO supervision stayed active instead of starving out:
  - step `306`
  - `self_distillation/success_group_fraction = 1.0`
  - `self_distillation/reprompt_sample_fraction = 0.99609375`
  - `self_distillation/empty_target_batch = 0.00390625`
- Generations stayed compact and well-formed rather than growing into the multi-thousand-token drift seen with OLMo.
- This is the strongest evidence so far that the OLMo result was dominated by the `Olmo3` runtime mismatch on `vllm 0.10.0`, not by a general SDPO FSDP failure on Section 3 Physics.

### [WIP] Run the Qwen Physics Megatron SDPO comparison

- The next run will reuse the same cluster and Physics launcher path but switch to `VARIANT=sdpo_megatron`.
- The comparison target is the healthy FSDP baseline above, keeping fixed:
  - `Qwen/Qwen3-8B`
  - SciKnowEval `physics`
  - validation every `5` steps
  - inline sample previews
  - capped validation dumps

### [Resolved] First Megatron launch failure came from unpushed local changes

- Job `7` failed with:
  - `Unknown VARIANT=sdpo_megatron`
  - `Expected one of: grpo_fsdp, sdpo_fsdp`
- Root cause:
  - the new Physics Megatron support existed only in the local workspace at first
  - the cluster always clones the fork branch from Git, so it picked up the older remote branch contents
- Fix:
  - committed the launcher/config changes as `81ee904a`
  - pushed `codex/sdpo-megatron-v070` to `https://github.com/zhoutong-hai/verl.git`

### [Resolved] Megatron Physics retry reached `main_ppo` and completed

- Retry job:
  - cluster `verl-olmo3-physics`
  - job `8`
  - W&B run `x4t8yyyi`
- It did not fail with a Megatron import/runtime traceback.
- Instead, it trained through the full schedule and then collapsed behaviorally:
  - the best validation point was already at step `5`
  - by the late run, it had fully entered the zero-target / one-token-output regime
- This means the Megatron Physics comparison is now usable:
  - FSDP SDPO: strong and sustained
  - Megatron SDPO: early promise, then collapse to zero

### [WIP] Megatron SDPO collapse appears tied to the current token-level reverse-KL design

- The healthy FSDP and collapsed Megatron runs used the same:
  - `Qwen/Qwen3-8B`
  - SciKnowEval `physics`
  - rollout counts, validation frequency, and reward function
- The strongest divergence is in the SDPO implementation itself:
  - FSDP path:
    - `teacher_scoring_mode=actor_worker`
    - `full_logit_distillation=true`
    - `alpha=0.5`
  - Megatron path:
    - `teacher_scoring_mode=trainer_ref`
    - `full_logit_distillation=false`
    - `alpha=1.0`
- Empirical comparison from the node logs:
  - FSDP step `20`:
    - `val-core/sciknoweval/acc/mean@16 = 0.6140625`
    - `response_length/mean = 448.0390625`
    - `self_distillation/reprompt_sample_fraction = 0.89453125`
    - `self_distillation/empty_target_batch = 0.10546875`
  - Megatron step `20`:
    - `val-core/sciknoweval/acc/mean@16 = 0.4375`
    - `response_length/mean = 2.0`
    - `self_distillation/reprompt_sample_fraction = 0.43359375`
    - `self_distillation/empty_target_batch = 0.56640625`
- Megatron behavior progression is very clear in saved validation dumps:
  - step `5`: still structured enough to get good answers
  - step `20`: outputs collapse to bare single-letter answers like `B`
  - step `200`: outputs are empty strings
- One especially suspicious metric difference:
  - FSDP early SDPO loss is positive
    - step `1`: `self_distillation/per_token_loss_mean = 0.0966`
  - Megatron early SDPO loss is negative
    - step `1`: `self_distillation/per_token_loss_mean = -0.1002`
    - step `5`: `self_distillation/per_token_loss_mean = -0.1059`
- The current code path lines up with that behavioral difference:
  - healthy FSDP path calls `compute_self_distillation_loss()` with on-the-fly teacher forwards from inside the actor worker, using full-logit distillation when configured:
    - [dp_actor.py](/Users/zhoutong/code/verl/verl/workers/actor/dp_actor.py#L844)
  - Megatron hard-restricts SDPO to token-level teacher log-probs only:
    - [megatron_actor.py](/Users/zhoutong/code/verl/verl/workers/actor/megatron_actor.py#L505)
    - [core_algos.py](/Users/zhoutong/code/verl/verl/trainer/ppo/core_algos.py#L909)
- The saved validation generations make the Megatron shortcut very explicit:
  - step `5`: still full `<reasoning> ... </reasoning><answer> ... </answer>` outputs
  - step `20`: many outputs are just bare single letters like `B`
  - step `200`: outputs are empty strings
- The current MCQ reward function makes that shortcut easier than intended:
  - [mcq.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/mcq.py#L4) extracts the answer by splitting on `<answer>`, so a plain `B` is still parsed as prediction `B`
  - [mcq.py](/Users/zhoutong/code/verl/verl/utils/reward_score/feedback/mcq.py#L25) assigns reward from that parsed letter even when the XML format is missing
  - result: Megatron can move from full structured answers to bare correct letters without losing reward immediately
- Current best diagnosis:
  - this does not look like a simple crash bug or bad tokenization issue
  - it looks like the current Megatron SDPO approximation is much more mode-seeking than the FSDP full-logit path
  - on this task, that combines badly with a reward parser that still accepts bare-letter answers
  - the likely collapse path is:
    - structured correct answers
    - bare-letter shortcut answers that still receive reward
    - EOS / empty-output collapse
    - no successful siblings
    - zero-target SDPO stall
- Most likely next fixes to test:
  - tighten the reward parser so only strict XML-formatted answers receive reward
  - or add a formatting penalty gate before a sample can count as a successful SDPO teacher
  - and, separately, move Megatron closer to the FSDP path by supporting top-k full-logit SDPO rather than the current token-level reverse-KL approximation

### [Resolved] Phase 1 Megatron full-logit top-k implementation is now in the branch

- A new Megatron `trainer_ref` SDPO path is now implemented locally to move beyond sampled-token distillation.
- The new path keeps the scalable Megatron layout and changes the teacher target payload:
  - old Megatron path:
    - `teacher_log_probs`
  - new Megatron path:
    - `teacher_log_probs`
    - `teacher_topk_indices`
    - `teacher_topk_log_probs`
- Implementation summary:
  - `megatron_workers.py`
    - adds a ref-worker API to emit compressed top-k teacher targets
  - `ray_trainer.py`
    - unions those top-k teacher targets into the SDPO batch for `teacher_scoring_mode=trainer_ref`
    - guards the path with `ppo_epochs == 1`
  - `megatron_actor.py`
    - gathers student log-probs on the cached teacher top-k support
    - calls the shared full-logit SDPO loss path instead of the old sampled-token-only path
- The implementation now uses TP-global vocabulary support when computing teacher/student top-k targets, rather than shard-local top-k.
- Updated Megatron SDPO configs now use:
  - `full_logit_distillation = true`
  - `distillation_topk = 100`
  - `alpha = 0.5`
- Local verification completed:
  - `python3 -m compileall` passed on the touched Python files
  - YAML parsing passed on the touched Megatron trainer configs
- Remaining work:
  - rerun the Qwen Physics Megatron job on this new path
  - check whether the earlier shortcut collapse (`structured -> bare letter -> empty output`) is reduced or eliminated
- That sign difference happens even though both runs log similar perplexity scale in the early phase, which suggests the current Megatron SDPO path is not semantically matching the FSDP loss path on the same task.
- Current working diagnosis:
  - the present Megatron implementation is not just "weaker than FSDP"; it is optimizing a materially different and less stable objective
  - once the EMA teacher falls behind, the token-level reverse-KL approximation appears to push the model toward ultra-short / empty outputs instead of stable distillation
  - this is consistent with the design gap already noted in `MEGATRON_SDPO_REVIEW.md`
