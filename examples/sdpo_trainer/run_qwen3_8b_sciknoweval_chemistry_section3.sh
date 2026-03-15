#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${VARIANT:-grpo_fsdp}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
CHEMISTRY_DATA_DIR="${CHEMISTRY_DATA_DIR:-/Users/zhoutong/code/SDPO/datasets/sciknoweval/chemistry}"
CHEMISTRY_TRAIN_FILE="${CHEMISTRY_TRAIN_FILE:-$CHEMISTRY_DATA_DIR/train.parquet}"
CHEMISTRY_VAL_FILE="${CHEMISTRY_VAL_FILE:-$CHEMISTRY_DATA_DIR/test.parquet}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-2}"
TRAIN_PP_SIZE="${TRAIN_PP_SIZE:-1}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_PP_SIZE="${ROLLOUT_PP_SIZE:-1}"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"

if [[ "$VARIANT" == "sdpo_megatron" ]]; then
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
fi

if [[ ! -f "$CHEMISTRY_TRAIN_FILE" || ! -f "$CHEMISTRY_VAL_FILE" ]]; then
  echo "Preprocessed parquet not found. Generating from $CHEMISTRY_DATA_DIR ..."
  python3 "$REPO_ROOT/examples/data_preprocess/sdpo_generalization.py" \
    --data_source "$CHEMISTRY_DATA_DIR"
fi

case "$VARIANT" in
  grpo_fsdp)
    CONFIG_NAME="grpo_fsdp_sciknoweval_chemistry_trainer.yaml"
    DEFAULT_EXP_NAME="qwen3_8b_section3_chemistry_grpo_fsdp"
    ;;
  sdpo_fsdp)
    CONFIG_NAME="sdpo_fsdp_sciknoweval_chemistry_trainer.yaml"
    DEFAULT_EXP_NAME="qwen3_8b_section3_chemistry_sdpo_fsdp"
    ;;
  sdpo_megatron)
    CONFIG_NAME="sdpo_megatron_sciknoweval_chemistry_trainer.yaml"
    DEFAULT_EXP_NAME="qwen3_8b_section3_chemistry_sdpo_megatron"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT"
    echo "Expected one of: grpo_fsdp, sdpo_fsdp, sdpo_megatron"
    exit 1
    ;;
esac

EXP_NAME="${EXP_NAME:-$DEFAULT_EXP_NAME}"

export MODEL_PATH
export CHEMISTRY_TRAIN_FILE
export CHEMISTRY_VAL_FILE
export N_GPUS_PER_NODE
export NNODES
export TRAIN_TP_SIZE
export TRAIN_PP_SIZE
export ROLLOUT_TP_SIZE
export ROLLOUT_PP_SIZE

python3 -m verl.trainer.main_ppo \
  --config-name "$CONFIG_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  "$@"
