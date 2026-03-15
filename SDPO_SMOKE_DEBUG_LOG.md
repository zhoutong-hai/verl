# SDPO Smoke Debug Log

Last updated: 2026-03-14

## Goal

Get the SkyPilot Megatron smoke run in
`examples/skypilot/verl-sdpo-megatron-smoke-qwen05b.yaml`
to launch cleanly enough to validate the SDPO path on the small public setup.

## Original Task

- Run and debug the SkyPilot Megatron smoke experiment end to end.
- Keep a repo-local log of issues found and how they were resolved.
- Push repo code changes to the fork branch when remote clone-based runs need them.
- Use the shared launch command as the canonical way to reproduce the experiment.

## Working Plan

1. Inspect the latest failure carefully before patching.
2. Patch the relevant code or launcher, update this debug log, and do a quick local verification when possible.
3. Commit and push repo code changes when the remote node needs to clone the updated branch.
4. Relaunch on SkyPilot, inspect the next outcome, and repeat until the smoke run succeeds or the next blocker is isolated clearly.

## Command

```bash
export WANDB_API_KEY='***'
source ../skypilot-infra/.venv/bin/activate
sky launch -c verl-sdpo-smoke /Users/zhoutong/code/verl/examples/skypilot/verl-sdpo-megatron-smoke-qwen05b.yaml --secret WANDB_API_KEY -y
```

## Current Status Snapshot

Update this section every time a task completes, a new issue is found, or an old issue is resolved.

- Last checked: 2026-03-14
- Cluster: `verl-sdpo-smoke`
- Latest remote task: `8`
- Branch / pushed commit: `codex/sdpo-megatron-v070` at `3c409cbc`
- Current run stage reached: task `8` got past setup, trainer initialization, self-distillation batch construction, and into the first actor update
- Current active blocker:
  Megatron actor SDPO loss path still sees `self_distillation` as a plain dict during the first actor update
- Plan status:
  1. Inspect the latest failure carefully before patching. `completed` for task `7`
  2. Patch the relevant code or launcher, update this debug log, and do a quick local verification when possible. `completed`
  3. Commit and push repo code changes when the remote node needs to clone the updated branch. `completed`
  4. Relaunch on SkyPilot, inspect the next outcome, and repeat until the smoke run succeeds or the next blocker is isolated clearly. `in_progress`
- Recent completed milestones:
  - setup passes on the smoke cluster
  - custom Megatron ref build no longer crashes on colocated actor+ref init
  - run now reaches `trainer.fit()` and SDPO batch construction
  - trainer-side SDPO config node is now normalized through `SelfDistillationConfig` before template access
  - trainer-side normalization fix pushed in `3c409cbc`
  - task `8` launched to validate the fix
  - task `8` got past the old `solution_template` crash
  - task `8` reached the first actor update before failing
- Local workspace state:
  - uncommitted changes in `SDPO_SMOKE_DEBUG_LOG.md`
  - uncommitted changes in `verl/workers/actor/megatron_actor.py`

## Debug Notes

### [Resolved] 2026-03-14: Initial setup-stage failures

Issue:
- the job originally failed in `setup`
- the launcher expected model/data at `/hai/zhoutong/...`
- the pod did not have `/hai` mounted, so the final `test -f ...` validation failed

Resolution:
- switched the launcher to `infra: k8s/hai-training`
- added the `/hai` PVC mount block under `config.kubernetes.pod_config`

Status:
- setup now passes

### [Resolved] 2026-03-14: Current run-stage failure

Issue:
- run now fails during Megatron worker model initialization
- fatal import error:
  `ModuleNotFoundError: No module named 'megatron.core.distributed.custom_fsdp'`

Where it happens:
- `verl/workers/megatron_workers.py`
- current path enters the vanilla `mbridge` import branch

Observed environment on remote node:
- `megatron-core==0.15.0`
- `megatron-bridge==0.2.2`
- `transformer-engine==2.2.0`
- `import mbridge` fails because it expects `megatron.core.distributed.custom_fsdp`
- `import megatron.bridge` succeeds

Current hypothesis:
- the smoke launcher should match the larger working launcher and set
  `actor_rollout_ref.actor.megatron.vanilla_mbridge=False`
  so the run uses `megatron.bridge` instead of vanilla `mbridge`

Next step:
- patch the smoke launcher to disable vanilla mbridge
- relaunch and inspect the next failure, if any

### [Resolved] 2026-03-14: Current rerun hang at Ray worker startup

Issue:
- a later rerun passed setup and got into `main_ppo`, but Ray workers failed to start
- fatal worker-side error:
  `FileNotFoundError: [Errno 2] No such file or directory`
  from `ray._private.runtime_env.working_dir.py` at `os.getcwd()`

What we found on the node:
- the custom training Ray cluster on port `6379` was still alive from a previous run
- its `raylet` cwd pointed to:
  `/root/verl-sdpo-smoke (deleted)`
- the launcher `setup` phase does `rm -rf "$VERL_REPO_DIR"` before recloning, so reruns
  deleted the cwd of the already-running custom Ray cluster
- the next `main_ppo` run reused that stale `6379` cluster, and new Ray workers crashed
  when inheriting the deleted cwd

Resolution applied:
- in `setup`, kill stale custom Ray processes under `/tmp/ray/session_` before deleting
  the repo checkout
- in `run`, start the custom `6379` Ray cluster from `/root` instead of from the repo dir
- export `RAY_ADDRESS=127.0.0.1:6379` so the trainer targets the custom cluster explicitly

Additional fix folded into the same launcher patch:
- set
  `actor_rollout_ref.actor.megatron.use_mbridge=True`
  and
  `actor_rollout_ref.actor.megatron.vanilla_mbridge=False`
  so the smoke run uses `megatron.bridge` instead of the incompatible vanilla `mbridge`

Next step:
- relaunch with the patched launcher
- verify that Ray workers start cleanly and the run gets past trainer initialization

### [Resolved] 2026-03-14: Ref model init bug in colocated Megatron SDPO worker

Issue:
- after the launcher fixes, the smoke run got through setup, Ray startup, dataset loading,
  and into Megatron worker initialization
- it then failed in `actor_rollout_ref_init_model` with:
  `AttributeError: 'NoneType' object has no attribute 'optimizer'`

Where it happens:
- `verl/workers/megatron_workers.py`
- `ActorRolloutRefWorker.init_model()` calls `_build_model_optimizer(..., optim_config=None)`
  to build the ref/teacher model
- inside `_build_model_optimizer()`, the colocated worker still takes the actor branch because
  `self._is_actor` is true for the combined actor+ref worker
- that falls through to `init_megatron_optim_config(optim_config=None, ...)`, which dereferences
  `optim_config.optimizer`

Why this is SDPO-specific:
- SDPO forces the `ActorRolloutRef` path even without KL
- that means actor and ref are initialized inside the same Megatron worker, so helper logic cannot
  rely only on worker-wide flags to decide whether it is building the actor or the teacher

Resolution applied:
- added an explicit `build_ref: bool = False` argument to `_build_model_optimizer()`
- when `build_ref=True`, the helper now:
  - selects `self.config.ref.megatron`
  - builds the ref module with `wrap_with_ddp=False`
  - loads ref weights
  - returns early without touching optimizer initialization
- `init_model()` now calls `_build_model_optimizer(..., build_ref=True)` for the ref/teacher

Status:
- local patch applied and syntax-checked
- this change must be committed and pushed before the next SkyPilot relaunch, because the
  launcher clones the fork branch on the remote node

### [Resolved] 2026-03-14: Custom Ray head readiness race in launcher

Issue:
- after pushing the ref-build fix, the next relaunch got through setup and started `main_ppo`
- it then failed almost immediately with:
  `Failed to connect to GCS at address 10.0.127.4:6379 within 5 seconds`

What happened:
- the launcher starts a custom system-Ray head on `127.0.0.1:6379`
- it then launches `python3 -m verl.trainer.main_ppo` right away
- `main_ppo` now correctly honors `RAY_ADDRESS=127.0.0.1:6379`, but it can race the new Ray
  head before GCS is fully ready

Resolution applied:
- in the head-node branch of the SkyPilot launcher, wait until `ray status` succeeds before
  launching the trainer

Status:
- local launcher patch applied
- no repo code push needed for this step because the YAML itself is launched from the local checkout

### [Resolved] 2026-03-14: Ref offload init bug in colocated Megatron SDPO worker

Issue:
- after the ref-build fix, the smoke run got deeper into `actor_rollout_ref_init_model`
- it then failed with:
  `AttributeError: 'AsyncActorRolloutRefWorker' object has no attribute '_ref_is_offload_param'`

Where it happens:
- `verl/workers/megatron_workers.py`
- `ActorRolloutRefWorker.init_model()` reaches:
  `if self._ref_is_offload_param:`
- the combined `actor_rollout_ref` worker sets both `self._is_actor` and `self._is_ref`
- but worker initialization still used `if self._is_actor ... elif self._is_ref ...`
  in the config-normalization block, so the ref-side branch never ran in the colocated case

Why this matters beyond one missing attribute:
- the same colocated pattern also meant the ref-side micro-batch normalization was skipped
- and `init_model()` reused actor transformer overrides when building the ref/teacher model

Resolution applied:
- initialize `self._ref_is_offload_param = False` unconditionally before normalization
- change the ref normalization block from `elif self._is_ref` to `if self._is_ref`
  so actor-side and ref-side setup both run in `actor_rollout_ref`
- split actor and ref transformer overrides in `init_model()`
  so the ref build uses `config.ref.megatron.override_transformer_config`
  instead of inheriting actor-only overrides

Status:
- patch committed and pushed in `87b1bf41`
- task `7` got past this path and reached `trainer.fit()`
- no longer the active blocker

### [Resolved] 2026-03-14: Custom Ray health check still too weak on relaunch

Issue:
- task `6` did not immediately reproduce the previous Python exception
- instead, it stalled in the launcher at:
  `Waiting for custom Ray on 127.0.0.1:6379...`

What we found:
- the head-node run block still used a process-existence shortcut before `ray start`
- an older `gcs_server` on port `6379` was still present from a previous failed attempt
- that stale process was enough to skip `ray start`, but not healthy enough for
  `ray status --address=127.0.0.1:6379` to succeed

Resolution in progress:
- replace the `ps aux | grep 6379` shortcut with a real
  `ray status --address="$RAY_ADDRESS"` health check
- if the health check fails, kill stale `/tmp/ray/session_` processes in the run block
  and start a fresh custom Ray head before entering the readiness loop

Status:
- local launcher patch applied
- task `7` got past the old stalled readiness loop
- no longer the active blocker

### [Resolved] 2026-03-14: SDPO config schema mismatch during teacher-batch construction

Issue:
- task `7` got through setup, custom Ray startup, Megatron worker initialization,
  and into the trainer loop
- it then failed in `trainer.fit()` with:
  `omegaconf.errors.ConfigAttributeError: Key 'solution_template' is not in struct`

Where it happens:
- `verl/trainer/ppo/ray_trainer.py`
- `_maybe_build_self_distillation_batch()`
- the code accesses:
  `actor_rollout_ref.actor.self_distillation.solution_template`

Observed failure:
- traceback points to:
  `self_distillation_cfg.solution_template.format(...)`
- OmegaConf says `solution_template` is not present in the structured config object

Current hypothesis:
- the trainer-side SDPO code expects config keys that are not declared in the
  `SelfDistillationConfig` dataclass used by the structured actor config
- either the config schema and the trainer implementation drifted apart, or the
  smoke config is still using the old/new field names inconsistently

Next step:
- inspect `verl/workers/config/actor.py` and the SDPO trainer config to compare the
  declared `self_distillation` fields against what `ray_trainer.py` reads
- patch either the schema or the trainer to make the field names consistent

Update:
- root cause is now clearer:
  `SelfDistillationConfig` defines `solution_template`, `feedback_template`, and
  `reprompt_template`, but `ray_trainer.py` was reading the raw Hydra config node,
  which only contained the explicitly overridden YAML keys
- local fix applied:
  normalize `actor_rollout_ref.actor.self_distillation` through
  `omega_conf_to_dataclass(..., dataclass_type=SelfDistillationConfig)`
  before the trainer reads template fields
- local verification:
  `python3 -m compileall verl/trainer/ppo/ray_trainer.py` passed
- remaining work:
  commit and push this patch, then relaunch to verify that task `8` gets past
  self-distillation batch construction

Status:
- patch committed and pushed in `3c409cbc`
- task `8` got past self-distillation batch construction and no longer reproduces the
  `solution_template` config error
- no longer the active blocker

### [WIP] 2026-03-15: Megatron actor SDPO loss still receives nested config as dict

Issue:
- task `8` got past the trainer-side SDPO config mismatch and reached the first actor update
- it then failed in the Megatron actor loss path with:
  `AttributeError: 'dict' object has no attribute 'full_logit_distillation'`

Where it happens:
- `verl/workers/actor/megatron_actor.py`
- inside the SDPO branch of `loss_func`
- failing line:
  `if self_distillation_cfg.full_logit_distillation:`

Root cause:
- `MegatronPPOActor` receives an instantiated `ActorConfig`, but the nested
  `self_distillation` field is still a plain dict rather than a fully normalized
  `SelfDistillationConfig`
- that means the trainer-side normalization fixed the batch-construction path, but the
  first actor update still crashes when it expects attribute access on the nested config

Resolution in progress:
- normalize `self.config.self_distillation` through
  `omega_conf_to_dataclass(..., dataclass_type=SelfDistillationConfig)` inside the
  SDPO loss branch before reading `full_logit_distillation`

Status:
- local patch applied
- `python3 -m compileall verl/workers/actor/megatron_actor.py` passed
- next step is commit, push, and relaunch task `9`
