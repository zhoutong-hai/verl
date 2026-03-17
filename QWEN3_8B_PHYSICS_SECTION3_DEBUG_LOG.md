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
