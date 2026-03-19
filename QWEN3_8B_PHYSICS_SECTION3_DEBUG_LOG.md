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
- Benchmark run metadata:
  - W&B run id: `lgteso46`
  - W&B URL: <https://wandb.ai/hippocraticai/olmo3_7b_section3_physics/runs/lgteso46>
  - experiment name: `qwen3_8b_section3_physics_sdpo_fsdp_20260317_011603`
  - note: this Qwen FSDP run was logged under the reused project name `olmo3_7b_section3_physics`
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

### [Resolved Locally] Megatron full-logit rerun exposed a response-alignment bug in trainer_ref top-k support

- After the Phase 1 full-logit implementation was pushed and manually relaunched on `verl-olmo3-physics`, the new Megatron run moved past initialization and into the first actor update, but then failed in the actor worker.
- Remote failure signature from `/tmp/ray/session_latest/logs/worker-6730908144e2290b02b67ffabd591ff3d16e5ac03c7434c182010f1f-03000000-1539274.err`:
  - `RuntimeError: Size does not match at dimension 1 expected index [1, 8192, 100] to be no larger than self [1, 240, 151936] apart from dimension 2`
  - crash site:
    - `verl/workers/actor/megatron_actor.py`, inside `logits_processor`
    - `topk_logits = torch.gather(full_logits, dim=-1, index=current_topk_indices)`
- Diagnosis:
  - `teacher_topk_indices` coming from the ref worker is response-aligned and padded to `[bs, max_response_len, k]`
  - the training-time Megatron `logits_processor` is operating on packed active tokens, so its TP-gathered logits are sequence-aligned with only the active positions, e.g. `[bs, active_seq_len, vocab]`
  - the code incorrectly treated the padded response-aligned teacher support as if it were already aligned to the packed logits
- Local fix applied:
  - when the teacher support length does not match the packed-logit sequence length, gather only on the active response positions
  - build a response-aligned `student_topk_log_probs` tensor from those active positions
  - skip the extra `[:, -response_length - 1 : -1, :]` reslice when the forward path has already returned response-aligned top-k log-probs
- Local verification:
  - `python3 -m compileall verl/workers/actor/megatron_actor.py` passed
- Next step:
  - push this alignment fix
  - rerun the same manual Qwen Physics Megatron job on `verl-olmo3-physics`
  - verify it gets past the first actor update and emits `training/global_step`

### [In Progress] Post-fix Megatron rerun launched on commit `47520ec7`

- The response-alignment fix was committed as:
  - `47520ec7` `Fix Megatron response-aligned top-k gathering`
- The remote repo on `verl-olmo3-physics` was updated to the same commit.
- A fresh manual rerun was launched under:
  - `/root/sky_logs/manual-megatron-20260317_185859/run.log`
- Current status at the latest check:
  - the rerun is alive
  - no `Traceback`, `RuntimeError`, or `Unhandled error` has been written to `run.log`
  - the previous `torch.gather(... teacher_topk_indices ...)` size-mismatch crash has not reappeared
  - but the job has not yet emitted `training/global_step` or validation metrics either
- So the immediate shape-mismatch failure appears resolved, but runtime verification is still incomplete until the rerun reaches the first actor update and logs a step.

### [Current Status] Previous gather crash is fixed, but the rerun still fails on a response-token alignment check

- Latest rerun:
  - `/root/sky_logs/manual-megatron-20260317_185859/run.log`
- The earlier failure:
  - `expected index [1, 8192, 100] to be no larger than self [1, 240, 151936]`
  - no longer appears
- New failure from the same first actor update:
  - `ValueError: Packed Megatron logits and response_mask disagree on active response tokens: 106 vs 105.`
  - variants of the same off-by-one mismatch also appeared:
    - `88 vs 87`
    - `101 vs 100`
- Crash site:
  - `verl/workers/actor/megatron_actor.py`, inside the new response-alignment branch in `logits_processor`
- Interpretation:
  - the major padded-vs-packed mismatch is resolved
  - but there is still a boundary/alignment bug between:
    - the packed Megatron sequence representation
    - the response-window view carried by `response_mask`
  - the mismatch is now consistently `1` token, which strongly suggests a shifted boundary issue rather than a general shape error
- Current job outcome:
  - no `training/global_step`
  - no validation metrics
  - the rerun dies during the first `actor_rollout_ref_update_actor()` call

### [Resolved Locally] Off-by-one came from Megatron `label_mask` including one extra token for short responses

- Root cause:
  - Megatron was building `label_mask` from the full `attention_mask` via:
    - keep positions `[-response_length - 1 : -1]`
    - clear the final sequence token
  - this only gives the right count when the response fully fills the padded response window
  - for shorter responses, it marks:
    - the prompt token immediately before the response
    - every valid response token
  - which is `r + 1` active positions for a response with `r` valid tokens
- Why the mismatch showed up as `106 vs 105`:
  - `response_mask.sum()` counts the real response tokens
  - `label_mask.sum()` counted one extra prediction slot whenever the response ended before the padded window ended
- Local fix applied in `megatron_actor.py`:
  - construct `label_mask` as an explicit one-token-left shift of `response_mask`
  - this gives exactly one active prediction position per real response token, even when the response is shorter than the padded response window
- Expected effect:
  - packed active logits and response-aligned teacher top-k support should now match exactly
  - the first Megatron actor update should get past the previous off-by-one assertion

### [Resolved Locally] Ref-worker teacher scoring needed a `response_mask` fallback

- After the off-by-one fix, the next rerun moved further and failed in the teacher ref-worker path instead of the first packed-logit alignment check.
- Remote failure from:
  - `/root/sky_logs/manual-megatron-20260317_201816/run.log`
- New traceback:
  - `KeyError: 'key "response_mask" not found in TensorDict with keys ['attention_mask', 'input_ids', 'position_ids', 'responses']'`
  - crash site:
    - `verl/workers/actor/megatron_actor.py`
    - in `compute_log_prob -> forward_step`
- Diagnosis:
  - the Megatron SDPO `label_mask` logic now depends on `response_mask`
  - the actor-update path already includes `response_mask` in the selected minibatch keys
  - but the ref-worker teacher-scoring path calls `compute_log_prob()` with only:
    - `input_ids`
    - `attention_mask`
    - `position_ids`
    - `responses`
  - so the ref worker needs to reconstruct `response_mask` when it is absent
- Local fix applied:
  - if `response_mask` is not present in the Megatron batch, derive it as:
    - `attention_mask[:, -response_length:]`
  - then build the shifted `label_mask` from that fallback
- Expected effect:
  - the Megatron ref-worker can now compute teacher top-k distillation targets without requiring `response_mask` to be explicitly carried in that path

### [In Progress] Latest Megatron rerun is live past the previous two crash points

- Remote repo on `verl-olmo3-physics` is now at:
  - `4a0e76f8`
- Latest manual rerun:
  - `/root/sky_logs/manual-megatron-20260317_202240/run.log`
- Current status:
  - `python3 -m verl.trainer.main_ppo` is still alive
  - the run is past startup and deep into rollout / CUDA-graph warmup
  - neither of the two most recent failure signatures has reappeared:
    - `Packed Megatron logits and response_mask disagree on active response tokens: ...`
    - `KeyError: 'response_mask'`
- So the latest rerun is materially healthier than the previous two, but it has not yet emitted the first `training/global_step` or validation checkpoint at the time of this update.

### [Current Status] Megatron rerun has now reached real training steps

- Live run:
  - `/root/sky_logs/manual-megatron-20260317_202240/run.log`
- The run has now emitted real training progress and is no longer failing during teacher-target construction or the first actor update.
- First observed steps:
  - step `1`
    - `training/global_step = 1`
    - `critic/score/mean = 0.6172`
    - `response_length/mean = 582.56`
    - `self_distillation/success_group_fraction = 0.8438`
    - `self_distillation/reprompt_sample_fraction = 0.8359`
    - `self_distillation/empty_target_batch = 0.1641`
  - step `2`
    - `training/global_step = 2`
    - `critic/score/mean = 0.5938`
    - `response_length/mean = 631.38`
    - `self_distillation/success_group_fraction = 0.8125`
    - `self_distillation/reprompt_sample_fraction = 0.7930`
    - `self_distillation/empty_target_batch = 0.2070`
- Important implication:
  - both recent Megatron fixes are now validated enough to show that:
    - the off-by-one alignment failure is gone
    - the missing `response_mask` failure in the ref-worker teacher-scoring path is gone
  - the next meaningful checkpoint is the first validation at step `5`

### [Resolved Locally] Packed-vs-response top-k tensor shape mismatch after step 11

- The latest rerun did not die from vanishing gradients. It trained through step `11` and logged:
  - `val-core/sciknoweval/acc/mean@16 = 0.596875`
  - `val-core/sciknoweval/acc/best@16/mean = 0.8138625`
  - `val-core/sciknoweval/acc/maj@16/mean = 0.61215`
- It then crashed in `actor_rollout_ref_update_actor()` with:
  - `RuntimeError: shape mismatch: value tensor of shape [8192, 100] cannot be broadcast to indexing result of shape [8460, 100]`
- Crash path:
  - `verl/workers/actor/megatron_actor.py`
  - `verl/models/mcore/model_forward.py`
  - `verl/models/mcore/util.py::postprocess_packed_seqs`
- Diagnosis:
  - the Megatron SDPO branch was building `topk_log_probs` in response-aligned form (`[bs, response_len, k]`) inside `logits_processor`
  - but the MCore forward wrapper still assumes every returned tensor is packed sequence-aligned and runs `postprocess_packed_seqs(...)` on it
  - so the postprocess step tried to scatter a response-length tensor into a packed sequence-length slot
- Local fix applied:
  - keep the teacher lookup keyed by the response-aligned teacher support
  - but write the gathered student top-k log-probs back into a packed sequence-aligned tensor at the active `label_mask` positions
  - do the same for `topk_indices` when requested
- Expected effect:
  - the MCore postprocess path and later Megatron loss slicing should now see consistent tensor layouts

### [Current Status] Latest Megatron rerun is alive but behaviorally collapsing by step 10

- Live run:
  - `/root/sky_logs/manual-megatron-20260317_204900/run.log`
- Current process is still alive and training:
  - `python3 -m verl.trainer.main_ppo`
- Early steps looked healthier than the previous run:
  - step `1`
    - `actor/grad_norm = 0.01005`
    - `self_distillation/success_group_fraction = 0.875`
    - `self_distillation/reprompt_sample_fraction = 0.8633`
    - `self_distillation/empty_target_batch = 0.1367`
    - `critic/score/mean = 0.6094`
    - `response_length/mean = 578.15`
- But by step `10`, the run had drifted badly:
  - `val-core/sciknoweval/acc/mean@16 = 0.0015625`
  - `val-core/sciknoweval/acc/best@16/mean = 0.0111375`
  - `val-core/sciknoweval/acc/maj@16/mean = 0.0017875`
  - `self_distillation/success_group_fraction = 0.21875`
  - `self_distillation/reprompt_sample_fraction = 0.203125`
  - `self_distillation/empty_target_batch = 0.796875`
  - `critic/score/mean = 0.09375`
  - `response_length/mean = 6813.53`
  - `response_length/clip_ratio = 0.8125`
  - `actor/grad_norm = 0.01162`
- So the latest Megatron run is not technically crashed, but it is clearly in a bad behavioral regime:
  - gradients are non-zero
  - SDPO supervision is mostly gone
  - outputs are running to the max-length cap
  - validation has effectively collapsed

### [Observed Failure Mode] Step-5 samples are normal, step-10 samples turn into repeated markdown image spam

- Validation dump paths:
  - step `5`:
    - `/hai/zhoutong/section3_physics_assets/validation_generations/qwen3_8b_section3_physics_sdpo_megatron/5.jsonl`
  - step `10`:
    - `/hai/zhoutong/section3_physics_assets/validation_generations/qwen3_8b_section3_physics_sdpo_megatron/10.jsonl`
- Step `5` sample quality:
  - `32/32` outputs include `<answer>...</answer>`
  - mean output length is about `918` characters
  - first samples are coherent physics solutions with correct predictions like `B`
- Step `10` sample quality:
  - `32/32` outputs contain repeated `![](https://i.imgur.com/...)` markdown image strings
  - `32/32` contain markdown image syntax
  - mean output length is about `18.3k` characters
  - `0/32` include `<answer>...</answer>`
  - example prediction becomes a long repeated Imgur markdown blob instead of `A/B/C/D`
- Input sanity check:
  - the underlying validation prompt is still a normal physics multiple-choice question
  - so this is not caused by corrupted evaluation data

### [Working Hypotheses] Most likely remaining Megatron issues

- The latest Megatron path is now technically stable enough to run, so the remaining issue is likely semantic rather than a simple crash bug.
- Highest-probability possibilities:
  - there is still a semantic mismatch between Megatron full-logit SDPO and the healthy FSDP path, even though the main tensor-shape crashes are fixed
  - the Megatron objective is still allowing a runaway long-output mode that FSDP does not enter on the same Qwen Physics setup
  - the trainer-ref teacher-target design may still be less stable than the FSDP actor-worker teacher path, even with `ppo_epochs = 1`
- Strong evidence for that:
  - Qwen Physics FSDP stayed healthy on the same task
  - the latest Megatron run does not die from zero gradients or immediate tensor bugs
  - instead it drifts into max-length repeated junk output while SDPO supervision supply collapses

### [In Progress] Added Megatron teacher-support diagnostics to test the loss-semantics hypothesis

- Added new Megatron-only SDPO debug metrics in [megatron_actor.py](/Users/zhoutong/code/verl/verl/workers/actor/megatron_actor.py):
  - `self_distillation/student_mass_on_teacher_support_mean`
  - `self_distillation/student_top1_in_teacher_support_fraction`
  - `self_distillation/student_top1_matches_teacher_top1_fraction`
- Why these matter:
  - the healthy FSDP path evaluates the teacher on the student's top-k support
  - the current Megatron `trainer_ref` path evaluates the student on the teacher's top-k support
  - if the student begins drifting toward bad tokens outside the teacher's support, the current Megatron path may only see that drift through the compressed tail bucket
- These new metrics are meant to show whether that is happening in practice:
  - low `student_mass_on_teacher_support_mean` would mean the teacher support is missing a lot of the student's actual probability mass
  - low `student_top1_in_teacher_support_fraction` would mean the student's top token is often outside the teacher's top-k set
  - low `student_top1_matches_teacher_top1_fraction` would mean the two policies are already choosing very different next-token modes
- `python3 -m compileall` passed after this instrumentation patch.
- The currently running Megatron job remained collapsed through steps `20` and `21`:
  - `success_group_fraction = 0.0`
  - `reprompt_sample_fraction = 0.0`
  - `empty_target_batch = 1.0`
  - `actor/pg_loss = 0.0`
  - `actor/grad_norm = 0.0`
  - `response_length/mean ~= 5.2k`
  - validation samples still show repeated markdown image spam
- Next action: relaunch the same Qwen Physics Megatron run with these new diagnostics enabled, then inspect whether the support-mismatch metrics collapse before or alongside output quality.

### [Resolved Locally] Diagnostic rerun hit metric-reduction failure before step 1

- The first diagnostic rerun (`manual-megatron-20260317_222033`) did not reach the first training step.
- Failure:
  - `ValueError: setting an array element with a sequence` in [reduce_metrics](/Users/zhoutong/code/verl/verl/utils/metric/utils.py)
- Interpretation:
  - one of the newly added debug metrics was being surfaced in a non-scalar form during actor metric aggregation
  - this was a logging/reduction robustness issue, not a new training-semantic failure
- Fix:
  - made [reduce_metrics](/Users/zhoutong/code/verl/verl/utils/metric/utils.py) coerce metric payloads into scalars before applying `mean` / `max` / `min`
  - this keeps the training loop robust even when debug instrumentation occasionally emits tensors, arrays, or short numeric sequences
- `python3 -m compileall` passed after the reducer patch.

### [Observed] Teacher-support metrics do not collapse ahead of the Megatron failure

- Diagnostic rerun:
  - `/root/sky_logs/manual-megatron-20260317_225010/run.log`
- New metrics stayed nearly flat from step `1` through step `10`:
  - `student_mass_on_teacher_support_mean`
    - step `1`: `0.0006688`
    - step `5`: `0.0006688`
    - step `10`: `0.0006706`
  - `student_top1_in_teacher_support_fraction`
    - step `1`: `0.9972`
    - step `5`: `0.9978`
    - step `10`: `1.0000`
  - `student_top1_matches_teacher_top1_fraction`
    - step `1`: `0.9351`
    - step `5`: `0.9388`
    - step `10`: `0.9433`
- Meanwhile the run still degraded sharply:
  - `success_group_fraction`
    - step `5`: `1.0`
    - step `10`: `0.3125`
  - `reprompt_sample_fraction`
    - step `5`: `0.9883`
    - step `10`: `0.2930`
  - `response_length/mean`
    - step `5`: `692.6`
    - step `10`: `6684.5`
  - `val-core/sciknoweval/acc/mean@16`
    - step `5`: `0.5805`
    - step `10`: `0.0008`
- Interpretation:
  - this weakens the simple version of the current hypothesis that "Megatron fails because the student's top tokens escape the teacher's cached top-k support"
  - the student top-1 token remains almost always inside teacher support, and top-1 agreement stays high, even while output quality collapses
- Important caveat:
  - `student_mass_on_teacher_support_mean` is suspiciously tiny but also almost perfectly flat from healthy to collapsed steps
  - so this metric may still need validation before using it as a strong standalone signal
- Current takeaway:
  - we have enough evidence to say the collapse is probably **not** primarily explained by simple student-top-token escape from teacher top-k support
  - the broader "Megatron loss semantics differ from healthy FSDP" concern still stands, but this particular mechanism is no longer the leading explanation

### [Completed] Step-1 FSDP vs Megatron SDPO debug dumps and offline tensor diff

- Collected step-`1` SDPO debug artifacts for both paths:
  - FSDP:
    - `/hai/zhoutong/section3_physics_assets/sdpo_debug_compare/fsdp_qwen/step_0001`
  - Megatron:
    - `/hai/zhoutong/section3_physics_assets/sdpo_debug_compare/megatron_qwen/step_0001`
- Each step directory now includes:
  - `trainer_summary.json`
  - `trainer_batch.pkl`
  - `teacher_batch.pkl`
  - `actor_pre_loss_rank0.pt`
  - Megatron also includes `teacher_targets.pkl`
- Offline comparison output:
  - `/hai/zhoutong/section3_physics_assets/sdpo_debug_compare/compare_fsdp_vs_megatron_step1.json`

#### Highest-signal compare results

- Actor-side metadata confirms the intended architecture split:
  - FSDP:
    - `support_source = student_topk`
    - `teacher_scoring_mode = actor_worker`
  - Megatron:
    - `support_source = teacher_topk`
    - `teacher_scoring_mode = trainer_ref`
- The pre-loss tensors are already different at step `1`:
  - `active_sdpo_tokens`
    - FSDP: `85`
    - Megatron: `92`
  - `response_mask_equal_fraction`: `0.9991455`
  - `student_log_probs_masked_mean_abs_diff`: `0.2560`
  - `teacher_log_probs_masked_mean_abs_diff`: `0.6799`
  - `support_index_exact_match_fraction`: `0.9887`
  - `support_set_exact_match_fraction`: `0.9886`
- Trainer-side teacher context also differs:
  - FSDP `teacher_input_ids` length: `11152`
  - Megatron `teacher_input_ids` length: `10380`

#### Interpretation

- This gives us the first concrete tensor-level evidence that the two implementations are **not** feeding the shared SDPO loss the same inputs, even at the first debug step.
- The largest mismatches are not the support indices themselves, which are close but not identical, but:
  - active SDPO token counts
  - masked student log-probs
  - masked teacher log-probs
  - teacher-context sequence length
- Important limitation:
  - this was a step-`1` compare across two separate runs, not a strict replay of one identical rollout batch through both implementations
  - so this is strong evidence of divergence, but not yet proof that any single tensor mismatch is the root cause
- Current practical takeaway:
  - the next highest-value debugging step is still a stricter replay-style compare on one shared rollout batch if we need to isolate the exact semantic delta
  - but the current diff already narrows the problem from "high-level training behavior" to "the two codepaths are constructing materially different SDPO supervision at loss time"

### [Resolved Locally] `skip_rollout` was not affecting async rollout runs

- The first attempt at a shared-rollout compare did not actually reuse rollout data.
- Root cause:
  - [RolloutSkip](/Users/zhoutong/code/verl/verl/utils/rollout_skip.py) only wrapped `actor_rollout_wg.generate_sequences()`
  - but these Physics runs use async rollout mode, so generation goes through `self.async_rollout_manager.generate_sequences(...)` in [ray_trainer.py](/Users/zhoutong/code/verl/verl/trainer/ppo/ray_trainer.py)
- Fix:
  - patched [ray_trainer.py](/Users/zhoutong/code/verl/verl/trainer/ppo/ray_trainer.py) so `skip_rollout=true` wraps the async rollout manager when `self.async_rollout_mode` is enabled
- Commit:
  - `04598039` — `Enable rollout skip for async manager`

### [Completed] Shared-rollout replay compare removes most student-side drift and isolates a teacher-side gap

- Shared rollout cache:
  - `/hai/zhoutong/section3_physics_assets/sdpo_debug_compare/shared_rollout_qwen_step1_v2/shared_qwen_physics_step1_v2_shared_qwen_physics_GBS32__N8`
- FSDP replay dump:
  - `/hai/zhoutong/section3_physics_assets/sdpo_debug_compare/fsdp_qwen_sharedrollout_v2/step_0001`
- Megatron replay dump:
  - `/hai/zhoutong/section3_physics_assets/sdpo_debug_compare/megatron_qwen_sharedrollout_v2/step_0001`
- Compare output:
  - `/hai/zhoutong/section3_physics_assets/sdpo_debug_compare/compare_fsdp_vs_megatron_sharedrollout_v2_step1.json`
- Megatron log now explicitly shows cache reuse:
  - `[RolloutSkip()] Successfully load pre-generated data from ... shared_qwen_physics_step1_v2_shared_qwen_physics_GBS32__N8`

#### Highest-signal shared-rollout results

- Actor-side comparison on the dumped pre-loss sample is now much tighter on the student path:
  - `active_sdpo_tokens`
    - FSDP: `77`
    - Megatron: `77`
  - `response_mask_equal_fraction`: `1.0`
  - `student_log_probs_masked_mean_abs_diff`: `0.00413`
  - `support_index_exact_match_fraction`: `0.99098`
  - `support_set_exact_match_fraction`: `0.99048`
- But the teacher path still diverges sharply on that same shared-rollout compare:
  - `teacher_log_probs_masked_mean_abs_diff`: `0.85524`
  - FSDP `teacher_log_prob_mean`: `-0.27905`
  - Megatron `teacher_log_prob_mean`: `-0.87333`

#### Interpretation

- This is the strongest evidence so far that the remaining semantic problem is now concentrated on the teacher side, not the student side:
  - once both runs consume the same cached rollout
  - the student-side SDPO tensors nearly align
  - but teacher-side SDPO tensors still differ a lot
- Important caveat:
  - the trainer debug dumps are intentionally capped to `8` sequences, so the saved trainer/teacher batch subsets can still differ due to batch balancing and slicing
  - that means trainer-batch-level equality is still a weaker signal than the actor pre-loss compare
- Current best read:
  - rollout randomness is no longer the main confounder
  - simple student-forward mismatch is no longer the main issue
  - the next investigation should focus on how teacher supervision is constructed and scored in:
    - FSDP `actor_worker`
    - Megatron `trainer_ref`

### [Resolved Locally] Order-sensitive demo selection was the main source of the teacher-side shared-rollout gap

- Additional direct check on the matched actor sample in the shared-rollout replay:
  - before the fix:
    - the teacher prompt text diverged at the `Correct solution:` demonstration block
    - FSDP and Megatron were choosing different successful sibling demonstrations for the same response
  - after the fix:
    - matched sample teacher prompt lengths are identical: `343` vs `343`
    - matched sample teacher prompt text is exactly equal
- Root cause:
  - [ray_trainer.py](/Users/zhoutong/code/verl/verl/trainer/ppo/ray_trainer.py) built `success_by_uid` in current batch order
  - `_get_solution()` then picked `solution_idxs[0]`
  - because `_balance_batch()` can reorder the same rollout differently across FSDP and Megatron, the chosen successful sibling demonstration could diverge across backends
- Fix:
  - sort successful sibling candidates deterministically before selecting the demonstration
  - current rule sorts by:
    - higher sequence score first
    - shorter response first
    - response text as a stable final key
- Commit:
  - `a7556680` — `Make SDPO demo selection deterministic`

#### Shared-rollout compare after deterministic demo selection

- Compare output:
  - `/hai/zhoutong/section3_physics_assets/sdpo_debug_compare/compare_fsdp_vs_megatron_sharedrollout_v3_step1.json`
- Teacher-side actor diff dropped sharply:
  - before:
    - `teacher_log_probs_masked_mean_abs_diff = 0.85524`
  - after:
    - `teacher_log_probs_masked_mean_abs_diff = 0.01855`
- Teacher prompt shapes also aligned:
  - FSDP `teacher_input_ids`: `10260`
  - Megatron `teacher_input_ids`: `10260`
- Student-side tensors remained close:
  - `response_mask_equal_fraction = 1.0`
  - `student_log_probs_masked_mean_abs_diff = 0.00413`
- Current interpretation:
  - the large teacher-side mismatch was primarily caused by order-sensitive demo selection, not by an inherent FSDP-vs-Megatron teacher-scoring mismatch
  - the remaining small differences now look much more like backend/scoring noise than a major semantic split

### [Concluded] Megatron full-logit/top-k path was reading logits after in-place cross-entropy mutation

- The next diagnostic hypothesis was whether Megatron's gathered full-logit path was internally inconsistent with the selected-token log-prob path.
- Instrumentation added in [megatron_actor.py](/Users/zhoutong/code/verl/verl/workers/actor/megatron_actor.py):
  - `self_distillation/selected_logprob_from_full_abs_diff_mean`
  - `self_distillation/selected_logprob_from_full_abs_diff_max`
  - `self_distillation/selected_logprob_from_full_fp32_abs_diff_mean`
  - `self_distillation/selected_logprob_from_full_fp32_abs_diff_max`
  - `self_distillation/student_top1_prob_from_full_fp32_mean`

#### Before fix: invariant was catastrophically broken at step 1

- Run:
  - `/root/sky_logs/manual-megatron-invariant-smoke.log`
- Step `1`:
  - `self_distillation/selected_logprob_from_full_abs_diff_mean = 10.79599`
  - `self_distillation/selected_logprob_from_full_abs_diff_max = 10.93123`
  - `self_distillation/selected_logprob_from_full_fp32_abs_diff_mean = 10.79599`
  - `self_distillation/student_top1_prob_mean = 1.6606e-05`
  - `self_distillation/student_top1_prob_from_full_fp32_mean = 1.6606e-05`
  - `training_ppl = 1.32246`
- Interpretation:
  - this ruled out a mere precision issue, because the fp32 reconstruction was equally wrong
  - the gathered full-logit/top-k path was not representing the same probabilities as `vocab_parallel_log_probs_from_logits(...)`

#### Root cause

- In Megatron-LM, `tensor_parallel.vocab_parallel_cross_entropy(...)` mutates its input shard logits in-place:
  - subtracts the max
  - exponentiates
  - normalizes to local softmax probabilities
- We were computing:
  - selected-token log-probs via `vocab_parallel_log_probs_from_logits(logits, label)`
  - then gathering `logits` afterward for the SDPO top-k/full-logit path
- So the top-k/full-logit path was often operating on already-mutated tensors instead of raw logits.
- Local Megatron reference:
  - [cross_entropy.py](/Users/zhoutong/code/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py)

#### Fix

- Patch in [megatron_actor.py](/Users/zhoutong/code/verl/verl/workers/actor/megatron_actor.py):
  - when `should_compute_topk` is true and the selected-token log-prob path would otherwise reuse `logits`,
  - clone shard logits before calling `vocab_parallel_log_probs_from_logits(...)`
  - preserve raw shard logits for the gathered full-logit/top-k path
- Commit:
  - `631b3a62` — `Preserve raw Megatron logits for SDPO top-k`

#### After fix: invariant is restored at step 1

- Run:
  - `/root/sky_logs/manual-megatron-invariant2-smoke.log`
- Step `1`:
  - `self_distillation/selected_logprob_from_full_abs_diff_mean = 5.54e-07`
  - `self_distillation/selected_logprob_from_full_abs_diff_max = 1.96e-06`
  - `self_distillation/selected_logprob_from_full_fp32_abs_diff_mean = 5.54e-07`
  - `self_distillation/student_top1_prob_mean = 0.9170`
  - `self_distillation/student_top1_prob_from_full_fp32_mean = 0.9170`
  - `self_distillation/student_mass_on_teacher_support_mean = 0.9969`
  - `training_ppl = 1.32246`
- This is the strongest implementation-level parity result so far:
  - the selected-token log-prob path and the gathered full-logit path now agree numerically
  - the earlier near-uniform top-k/support metrics were artifacts of the in-place mutation bug

#### Early post-fix training health

- Same fixed run through step `3` remains healthy:
  - step `2`
    - `response_length/mean = 620.6`
    - `success_group_fraction = 0.71875`
    - `actor/grad_norm = 0.3045`
    - `selected_logprob_from_full_abs_diff_mean = 5.76e-07`
  - step `3`
    - `response_length/mean = 627.5`
    - `success_group_fraction = 0.78125`
    - `actor/grad_norm = 0.3173`
    - `selected_logprob_from_full_abs_diff_mean = 5.64e-07`
- Step `4` also stays stable:
  - `response_length/mean = 586.9`
  - `success_group_fraction = 0.8125`
  - `actor/grad_norm = 0.2638`
  - `selected_logprob_from_full_abs_diff_mean = 5.54e-07`
- First post-fix validation at step `5` is healthy:
  - `val-core/sciknoweval/acc/mean@16 = 0.5875`
  - `val-core/sciknoweval/acc/best@16/mean = 0.83015`
  - `val-core/sciknoweval/acc/maj@16/mean = 0.63576`
  - `response_length/mean = 526.2`
  - `response_length/clip_ratio = 0.0`
  - `success_group_fraction = 1.0`
  - `reprompt_sample_fraction = 0.98828`
  - `empty_target_batch = 0.01172`
  - `incorrect_format/mean@16 = 0.99453`
  - `selected_logprob_from_full_abs_diff_mean = 5.37e-07`
- Step `10` stays healthy instead of collapsing:
  - `val-core/sciknoweval/acc/mean@16 = 0.58984`
  - `val-core/sciknoweval/acc/best@16/mean = 0.78666`
  - `val-core/sciknoweval/acc/maj@16/mean = 0.63728`
  - `response_length/mean = 423.1`
  - `response_length/clip_ratio = 0.0`
  - `success_group_fraction = 0.96875`
  - `reprompt_sample_fraction = 0.95703`
  - `empty_target_batch = 0.04297`
  - `incorrect_format/mean@16 = 0.98125`
  - `selected_logprob_from_full_abs_diff_mean = 5.27e-07`
- Step `15` improves further on validation:
  - `val-core/sciknoweval/acc/mean@16 = 0.65469`
  - `val-core/sciknoweval/acc/best@16/mean = 0.86953`
  - `val-core/sciknoweval/acc/maj@16/mean = 0.70779`
  - `response_length/mean = 426.2`
  - `response_length/clip_ratio = 0.0`
  - `success_group_fraction = 0.9375`
  - `reprompt_sample_fraction = 0.9375`
  - `empty_target_batch = 0.0625`
  - `incorrect_format/mean@16 = 0.97891`
  - `selected_logprob_from_full_abs_diff_mean = 5.31e-07`
- By step `17`, the run is still in the same healthy regime:
  - `response_length/mean = 371.9`
  - `response_length/clip_ratio = 0.0`
  - `success_group_fraction = 0.96875`
  - `reprompt_sample_fraction = 0.96484`
  - `empty_target_batch = 0.03516`
  - `actor/grad_norm = 0.23488`
  - `selected_logprob_from_full_abs_diff_mean = 5.08e-07`
- The full 20-step smoke run also finishes healthy:
  - step `20`
  - `val-core/sciknoweval/acc/mean@16 = 0.64688`
  - `val-core/sciknoweval/acc/best@16/mean = 0.85637`
  - `val-core/sciknoweval/acc/maj@16/mean = 0.66915`
  - `response_length/mean = 393.4`
  - `response_length/clip_ratio = 0.0`
  - `success_group_fraction = 0.90625`
  - `reprompt_sample_fraction = 0.90234`
  - `empty_target_batch = 0.09766`
  - `incorrect_format/mean@16 = 0.99531`
  - `selected_logprob_from_full_abs_diff_mean = 4.98e-07`

#### Updated conclusion after the corrected smoke run

- The old Megatron failure signature is gone on this corrected path:
  - no response-length explosion by step `10`
  - no SDPO target starvation by step `10`
  - no collapse to zero validation by step `10`
- The corrected Megatron run is healthy well past the old failure window:
  - step `15` validation is stronger than both step `5` and step `10`
  - the full 20-step smoke run finishes without entering the old collapse regime
  - the invariant remains numerically tight through the final observed step
- Current best conclusion:
  - the in-place cross-entropy mutation bug was a real correctness bug in Megatron full-logit SDPO
  - fixing it materially changed training behavior, not just diagnostics
  - this is strong evidence that the previous immediate Megatron collapse was implementation-driven
  - the corrected Megatron path now looks healthy enough to justify a full-length parity run
- Remaining caveat:
  - this still does **not** prove full long-run parity with FSDP
  - the next fair check is a longer corrected Megatron run and comparison of the best `1h` and `5h` windows against the healthy FSDP Physics baseline

### [Ongoing] Matched-step comparison across live Qwen Physics runs

- To avoid mixing `mean@16`, `best@16`, and `maj@16`, the cleanest comparison is:
  - `val-core/sciknoweval/acc/mean@16`
  - aligned at the same validation cadence (`test_freq = 5`)
- Current live runs:
  - Megatron SDPO `student_topk` on `verl-olmo3-physics`
  - FSDP SDPO repro on `verl-qwen3-physics-fsdp-repro`
  - FSDP GRPO repro on `verl-qwen3-physics-grpo-repro`

| Step | FSDP SDPO repro | Megatron SDPO `student_topk` | FSDP GRPO repro |
| --- | ---: | ---: | ---: |
| 5 | `0.565625` | `0.5640625` | `0.58359375` |
| 10 | `0.62890625` | `0.61875` | `0.61640625` |
| 20 | `0.640625` | `0.6515625` | `0.603125` |
| 40 | `0.6875` | `0.6796875` | `0.509375` |
| 60 | `0.70234375` | `0.69296875` | `0.6015625` |
| 80 | `0.74609375` | `0.7` | `0.61953125` |
| 90 | `0.70859375` | `0.70234375` | `0.515625` |

- Readout from the shared window so far:
  - the two SDPO runs are now in the same regime and track each other reasonably closely
  - FSDP SDPO is still slightly stronger overall in this matched-step view, with the clearest gap around step `80`
  - GRPO is clearly behind both SDPO runs and degrades later in the shared window

- Latest visible `val-core/sciknoweval/acc/mean@16` values beyond the shared window:
  - Megatron SDPO `student_topk`: `0.72734375`
  - FSDP SDPO repro: `0.7125`
  - FSDP GRPO repro: `0.515625`

- Current interpretation:
  - the catastrophic Megatron failure mode is resolved
  - the remaining Megatron-vs-FSDP question is now about quality parity, not correctness collapse
  - both SDPO variants are materially stronger than the current GRPO repro on this Qwen Physics setup
