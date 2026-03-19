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

## Current Status Snapshot (2026-03-16 16:28 PDT)

- The two OLMo FSDP runs are still active on the reused Chemistry clusters:
  - `verl-qwen3-chemistry-grpo` -> job `10` -> W&B `g2wwcwfx`
  - `verl-qwen3-chemistry-sdpo` -> job `8` -> W&B `7vohyks1`
- GRPO is still learning, but early validation is modest so far:
  - latest visible step: `30`
  - latest visible validation at step `30`:
    - `val-core/sciknoweval/acc/mean@16 = 0.0479`
    - `val-core/sciknoweval/acc/best@16/mean = 0.2662`
    - `val-core/sciknoweval/acc/maj@16/mean = 0.0659`
  - latest visible train metrics at step `30`:
    - `critic/score/mean = 0.0352`
    - `response_length/mean = 193.3`
    - `actor/entropy = 1.1599`
- SDPO is showing the same target-starvation failure mode seen in earlier Chemistry runs:
  - latest visible step: `24`
  - latest visible validation at step `20`:
    - `val-core/sciknoweval/acc/mean@16 = 0.0`
    - `val-core/sciknoweval/acc/best@16/mean = 0.0`
    - `val-core/sciknoweval/acc/maj@16/mean = 0.0`
  - latest visible train metrics at step `24`:
    - `critic/score/mean = 0.0`
    - `response_length/mean = 2531.0`
    - `self_distillation/success_group_fraction = 0.0`
    - `self_distillation/reprompt_sample_fraction = 0.0`
    - `self_distillation/active_token_fraction = 0.0`
    - `self_distillation/empty_target_batch = 1.0`
- The OLMo rollout path is still using vLLM's Transformers fallback, which may affect rollout speed and comparability with the paper environment.
- Dataset parity gap remains:
  - current local dataset path is still `/hai/zhoutong/section3_chemistry_assets/data/sciknoweval_chemistry`
  - the public paper-aligned OLMo run used `sciknoweval/chemistry_filtered`
- Future OLMo reruns will now persist validation samples explicitly instead of relying on reward-manager terminal prints:
  - `trainer.log_val_generations = 8`
  - `trainer.validation_data_dir = $VALIDATION_DATA_DIR`

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

### [Resolved] Reuse the existing Chemistry clusters for the OLMo runs

- The OLMo GRPO and SDPO jobs were launched onto the already-running Chemistry cluster names instead of provisioning new nodes:
  - `verl-qwen3-chemistry-grpo`
  - `verl-qwen3-chemistry-sdpo`
- Both reused-cluster jobs now have live run logs and W&B runs:
  - GRPO FSDP job `10` -> W&B `g2wwcwfx`
  - SDPO FSDP job `8` -> W&B `7vohyks1`

### [WIP] OLMo rollout path falls back to Transformers inside vLLM

- The live OLMo run logs show:
  - `Olmo3ForCausalLM has no vLLM implementation, falling back to Transformers implementation.`
- This is not a launch blocker, because both jobs still started training successfully.
- It is still important for interpretation because it may affect rollout throughput and fidelity relative to the paper environment.

### [WIP] OLMo SDPO has already fallen into the zero-target regime

- The current OLMo SDPO run started with non-zero SDPO activity:
  - step `1`: `success_group_fraction = 0.4063`, `reprompt_sample_fraction = 0.3555`
  - step `10`: `success_group_fraction = 0.0938`, `reprompt_sample_fraction = 0.0820`
- By step `15`, the run had already lost all successful sibling targets:
  - `success_group_fraction = 0.0`
  - `reprompt_sample_fraction = 0.0`
  - `active_token_fraction = 0.0`
  - `empty_target_batch = 1.0`
- The current latest visible state at step `24` is still fully stalled:
  - `actor/pg_loss = 0.0`
  - `actor/grad_norm = 0.0`
  - `critic/score/mean = 0.0`
  - `val-core/sciknoweval/acc/mean@16 = 0.0` at the latest visible validation step `20`
- So switching from Qwen to OLMo did not remove the current SDPO starvation pattern on the plain Chemistry dataset.

#### OLMo SDPO trajectory

| Step | `acc@16` | `best@16` | `success_group_fraction` | `reprompt_sample_fraction` | `active_token_fraction` | `critic/score/mean` | `response_length/mean` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | - | - | 0.4063 | 0.3555 | 0.3555 | 0.0508 | 1038.7 |
| 2 | - | - | 0.2812 | 0.2539 | 0.2539 | 0.0430 | 1065.3 |
| 3 | - | - | 0.2812 | 0.2500 | 0.2500 | 0.0391 | 1337.8 |
| 5 | 0.0372 | 0.2863 | 0.4063 | 0.3711 | 0.3711 | 0.0703 | 900.0 |
| 10 | 0.0071 | 0.0677 | 0.0938 | 0.0820 | 0.0820 | 0.0117 | 2311.5 |
| 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3429.8 |
| 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2874.1 |
| 24 | - | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2531.0 |

### [WIP] OLMo GRPO is alive, but early validation is modest and decaying after an initial rise

- The current OLMo GRPO run is still training and has not hard-stalled like SDPO.
- But its early validation curve is not monotonic:
  - step `10`: `acc@16 = 0.0643`, `best@16 = 0.4265`
  - step `15`: `acc@16 = 0.1003`, `best@16 = 0.5788`
  - step `20`: `acc@16 = 0.0890`, `best@16 = 0.5334`
  - step `25`: `acc@16 = 0.0518`, `best@16 = 0.2862`
  - step `30`: `acc@16 = 0.0479`, `best@16 = 0.2662`
- Unlike SDPO, it is still taking gradients and receiving non-zero rewards:
  - step `30`: `actor/grad_norm = 0.1324`, `critic/score/mean = 0.0352`
- So the current OLMo picture is:
  - GRPO is still training, but its early gains have softened
  - SDPO has already fully starved out

#### OLMo GRPO trajectory

| Step | `acc@16` | `best@16` | `maj@16` | `critic/score/mean` | `response_length/mean` | `actor/entropy` | `actor/grad_norm` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | - | - | - | 0.0352 | 1205.5 | 5.2233 | 0.0618 |
| 2 | - | - | - | 0.0195 | 1093.2 | 5.1341 | 0.0587 |
| 5 | 0.0330 | 0.2615 | 0.0471 | 0.0430 | 1006.5 | 5.2284 | 0.0854 |
| 10 | 0.0643 | 0.4265 | 0.1033 | 0.0703 | 971.5 | 4.7037 | 0.1016 |
| 15 | 0.1003 | 0.5788 | 0.1501 | 0.0898 | 1103.3 | 3.5424 | 0.1066 |
| 20 | 0.0890 | 0.5334 | 0.1224 | 0.1523 | 1307.8 | 2.6182 | 0.2206 |
| 25 | 0.0518 | 0.2862 | 0.0739 | 0.0430 | 362.3 | 1.7966 | 0.1277 |
| 30 | 0.0479 | 0.2662 | 0.0659 | 0.0352 | 193.3 | 1.1599 | 0.1324 |

### [Completed] Final outcome for the OLMo Chemistry attempt

- The Chemistry log is now complete enough to treat as a failed reproduction attempt with clear reasons.
- Observed outcome on the current public Chemistry setup:
  - SDPO FSDP (`7vohyks1`) starved into the zero-target regime and reached `acc@16 = 0.0`
  - GRPO FSDP (`g2wwcwfx`) stayed alive but remained weak and decayed after an early bump
- The two main unresolved parity gaps are still:
  - OLMo rollout uses the unsupported Transformers fallback under `vllm 0.10.0`
  - dataset parity is incomplete because the public run here uses `chemistry`, while the paper-aligned OLMo setup used `chemistry_filtered`
- Practical takeaway:
  - this Chemistry result should not be used as the final quality judgment of SDPO or Megatron
  - it is best treated as evidence that OLMo Chemistry reproduction remains blocked on runtime support and dataset parity, not on logging or launcher plumbing
