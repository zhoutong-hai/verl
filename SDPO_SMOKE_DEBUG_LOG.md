# SDPO Smoke Debug Log

Last updated: 2026-03-14

## Goal

Get the SkyPilot Megatron smoke run in
`examples/skypilot/verl-sdpo-megatron-smoke-qwen05b.yaml`
to launch cleanly enough to validate the SDPO path on the small public setup.

## Command

```bash
export WANDB_API_KEY='***'
source ../skypilot-infra/.venv/bin/activate
sky launch -c verl-sdpo-smoke /Users/zhoutong/code/verl/examples/skypilot/verl-sdpo-megatron-smoke-qwen05b.yaml --secret WANDB_API_KEY -y
```

## Debug Notes

### 2026-03-14: Initial setup-stage failures

Issue:
- the job originally failed in `setup`
- the launcher expected model/data at `/hai/zhoutong/...`
- the pod did not have `/hai` mounted, so the final `test -f ...` validation failed

Resolution:
- switched the launcher to `infra: k8s/hai-training`
- added the `/hai` PVC mount block under `config.kubernetes.pod_config`

Status:
- setup now passes

### 2026-03-14: Current run-stage failure

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

### 2026-03-14: Current rerun hang at Ray worker startup

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

### 2026-03-14: Ref model init bug in colocated Megatron SDPO worker

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

### 2026-03-14: Custom Ray head readiness race in launcher

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

### 2026-03-14: Ref offload init bug in colocated Megatron SDPO worker

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
- local patch applied and syntax-checked
- this change must be committed and pushed before the next SkyPilot relaunch, because the
  launcher clones the fork branch on the remote node
