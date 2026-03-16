#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${VARIANT:-grpo_fsdp}"
REMOTE_MODEL_PATH="/hai/zhoutong/section3_chemistry_assets/models/Olmo-3-7B-Instruct"
REMOTE_CHEMISTRY_DATA_DIR="/hai/zhoutong/section3_chemistry_assets/data/sciknoweval_chemistry"
LOCAL_CHEMISTRY_DATA_DIR="/Users/zhoutong/code/SDPO/datasets/sciknoweval/chemistry"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -f "$REMOTE_MODEL_PATH/config.json" ]]; then
    MODEL_PATH="$REMOTE_MODEL_PATH"
  else
    MODEL_PATH="allenai/Olmo-3-7B-Instruct"
  fi
fi

if [[ -z "${CHEMISTRY_DATA_DIR:-}" ]]; then
  if [[ -f "$REMOTE_CHEMISTRY_DATA_DIR/train.json" && -f "$REMOTE_CHEMISTRY_DATA_DIR/test.json" ]]; then
    CHEMISTRY_DATA_DIR="$REMOTE_CHEMISTRY_DATA_DIR"
  else
    CHEMISTRY_DATA_DIR="$LOCAL_CHEMISTRY_DATA_DIR"
  fi
fi

CHEMISTRY_TRAIN_FILE="${CHEMISTRY_TRAIN_FILE:-$CHEMISTRY_DATA_DIR/train.parquet}"
CHEMISTRY_VAL_FILE="${CHEMISTRY_VAL_FILE:-$CHEMISTRY_DATA_DIR/test.parquet}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"

if [[ ! -f "$CHEMISTRY_TRAIN_FILE" || ! -f "$CHEMISTRY_VAL_FILE" ]]; then
  echo "Preprocessed parquet not found. Generating from $CHEMISTRY_DATA_DIR ..."
  python3 "$REPO_ROOT/examples/data_preprocess/sdpo_generalization.py" \
    --data_source "$CHEMISTRY_DATA_DIR"
fi

case "$VARIANT" in
  grpo_fsdp)
    CONFIG_NAME="grpo_fsdp_sciknoweval_chemistry_olmo_trainer.yaml"
    DEFAULT_EXP_NAME="olmo3_7b_section3_chemistry_grpo_fsdp"
    ;;
  sdpo_fsdp)
    CONFIG_NAME="sdpo_fsdp_sciknoweval_chemistry_olmo_trainer.yaml"
    DEFAULT_EXP_NAME="olmo3_7b_section3_chemistry_sdpo_fsdp"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT"
    echo "Expected one of: grpo_fsdp, sdpo_fsdp"
    exit 1
    ;;
esac

EXP_NAME="${EXP_NAME:-$DEFAULT_EXP_NAME}"

export MODEL_PATH
export CHEMISTRY_TRAIN_FILE
export CHEMISTRY_VAL_FILE
export N_GPUS_PER_NODE
export NNODES

python3 -m verl.trainer.main_ppo \
  --config-name "$CONFIG_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  "$@"
