#!/usr/bin/env bash

set -euo pipefail
set -x

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/test.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
VARIANT="${VARIANT:-sdpo}"

case "$VARIANT" in
  sdpo)
    CONFIG_NAME="sdpo_megatron_smoke_trainer.yaml"
    DEFAULT_EXP_NAME="qwen2_5_0_5b_gsm8k_sdpo_megatron_smoke"
    ;;
  grpo)
    CONFIG_NAME="grpo_megatron_smoke_trainer.yaml"
    DEFAULT_EXP_NAME="qwen2_5_0_5b_gsm8k_grpo_megatron_smoke"
    ;;
  *)
    echo "Unsupported VARIANT: $VARIANT"
    echo "Expected one of: sdpo, grpo"
    exit 1
    ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-$DEFAULT_EXP_NAME}"

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  python3 "$PROJECT_DIR/examples/data_preprocess/gsm8k.py" --local_save_dir "$DATA_DIR"
fi

python3 -m verl.trainer.main_ppo \
  --config-path="$PROJECT_DIR/verl/trainer/config" \
  --config-name="$CONFIG_NAME" \
  data.train_files="['$TRAIN_FILE']" \
  data.val_files="['$VAL_FILE']" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  "$@"
