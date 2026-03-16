# OLMo-3-7B Chemistry Section 3 Debug Log

## Goal

Reproduce the paper-style Chemistry Section 3 setup with `allenai/Olmo-3-7B-Instruct` on the current branch, focusing on the fair FSDP comparison:

1. on-policy GRPO baseline
2. upstream-style SDPO FSDP

## Original Task

Switch the Chemistry Section 3 experiment from Qwen to the paper's OLMo model and stage the model under `/hai/zhoutong` so the run no longer depends on the removed `model-eval` host.

## Working Plan

1. Stage `allenai/Olmo-3-7B-Instruct` into `/hai/zhoutong/section3_chemistry_assets/models/Olmo-3-7B-Instruct`.
2. Add OLMo-specific FSDP configs and a dedicated SkyPilot launcher.
3. Launch GRPO and SDPO FSDP runs with the same Chemistry dataset.
4. Track the same `1h` / `5h` validation windows used in the paper-style comparison.

## Commands

### SkyPilot: OLMo GRPO FSDP

```bash
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
export WANDB_API_KEY='***'
sky launch -c verl-olmo3-chemistry-grpo \
  /Users/zhoutong/code/verl/examples/skypilot/verl-olmo3-section3-chemistry.yaml \
  --env VARIANT=grpo_fsdp \
  --secret WANDB_API_KEY -y
```

### SkyPilot: OLMo SDPO FSDP

```bash
source /Users/zhoutong/code/skypilot-infra/.venv/bin/activate
export WANDB_API_KEY='***'
sky launch -c verl-olmo3-chemistry-sdpo \
  /Users/zhoutong/code/verl/examples/skypilot/verl-olmo3-section3-chemistry.yaml \
  --env VARIANT=sdpo_fsdp \
  --secret WANDB_API_KEY -y
```

## Monitoring Metrics

- `val-core/sciknoweval/acc/mean@16`
- `val-aux/sciknoweval/reward/mean@16`
- `self_distillation/reprompt_sample_fraction`
- `self_distillation/active_token_fraction`
- `self_distillation/empty_target_batch`
- `critic/score/mean`
- `response_length/mean`
- `response_length/clip_ratio`
- `actor/entropy`
- `perf/throughput`

## Current Status Snapshot (2026-03-16)

- OLMo-specific FSDP configs and a dedicated SkyPilot launcher have been added locally.
- The shared model path is now staged:
  - `/hai/zhoutong/section3_chemistry_assets/models/Olmo-3-7B-Instruct`
- `model-eval` is no longer reachable, so the checkpoint was downloaded from a live Chemistry SkyPilot cluster instead.
- The new OLMo launcher also includes a setup-time download fallback if that shared path is missing on a future cluster.
- Dataset parity gap remains:
  - current local dataset path is still `/hai/zhoutong/section3_chemistry_assets/data/sciknoweval_chemistry`
  - the public paper-aligned OLMo run used `sciknoweval/chemistry_filtered`

## Debug Notes

### [Resolved] Stage `allenai/Olmo-3-7B-Instruct` into shared `/hai/zhoutong`

- `model-eval` is no longer reachable.
- Workaround used: a live Chemistry SkyPilot cluster downloaded the model into `/hai/zhoutong/section3_chemistry_assets/models/Olmo-3-7B-Instruct`.
- The new OLMo SkyPilot launcher also performs the same download during `setup:` if the shared path is still empty.

### [Resolved] Avoid incompatible `huggingface_hub` upgrade in the OLMo launcher

- Initial OLMo launcher version upgraded `huggingface_hub` to `1.7.1`.
- That conflicts with the cluster image's `transformers==4.55.4` requirement of `huggingface-hub<1.0`.
- Final fix: explicitly restore a compatible package set in [verl-olmo3-section3-chemistry.yaml](/Users/zhoutong/code/verl/examples/skypilot/verl-olmo3-section3-chemistry.yaml):
  - `transformers==4.57.1`
  - `huggingface_hub==0.36.2`
- This matches the paper-era `transformers` version more closely and repairs reused clusters even if an earlier failed setup mutated the Python environment.

### [Resolved] Remove shell heredoc parsing bug from the OLMo launcher

- The first OLMo launcher used an indented shell heredoc inside the SkyPilot `setup:` block.
- On the remote cluster, that produced:
  - `warning: here-document ... wanted 'PY'`
  - `syntax error: unexpected end of file`
- Fix: replace the heredoc with a single-line `python3 -c ... snapshot_download(...)` call.
