#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${VARIANT:-grpo_fsdp}"
REMOTE_MODEL_PATH="/hai/zhoutong/section3_chemistry_assets/models/Olmo-3-7B-Instruct"
REMOTE_PHYSICS_DATA_DIR="/hai/zhoutong/section3_physics_assets/data/sciknoweval_physics"
REMOTE_VALIDATION_ROOT="/hai/zhoutong/section3_physics_assets/validation_generations"
LOCAL_PHYSICS_DATA_DIR="/Users/zhoutong/code/SDPO/datasets/sciknoweval/physics"
LOCAL_VALIDATION_ROOT="$REPO_ROOT/outputs/validation_generations"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -f "$REMOTE_MODEL_PATH/config.json" ]]; then
    MODEL_PATH="$REMOTE_MODEL_PATH"
  else
    MODEL_PATH="allenai/Olmo-3-7B-Instruct"
  fi
fi

if [[ -z "${PHYSICS_DATA_DIR:-}" ]]; then
  if [[ -f "$REMOTE_PHYSICS_DATA_DIR/train.json" && -f "$REMOTE_PHYSICS_DATA_DIR/test.json" ]]; then
    PHYSICS_DATA_DIR="$REMOTE_PHYSICS_DATA_DIR"
  else
    PHYSICS_DATA_DIR="$LOCAL_PHYSICS_DATA_DIR"
  fi
fi

PHYSICS_TRAIN_FILE="${PHYSICS_TRAIN_FILE:-$PHYSICS_DATA_DIR/train.parquet}"
PHYSICS_VAL_FILE="${PHYSICS_VAL_FILE:-$PHYSICS_DATA_DIR/test.parquet}"
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

if [[ ! -f "$PHYSICS_TRAIN_FILE" || ! -f "$PHYSICS_VAL_FILE" ]]; then
  echo "Preprocessed parquet not found. Generating from $PHYSICS_DATA_DIR ..."
  python3 "$REPO_ROOT/examples/data_preprocess/sdpo_generalization.py" \
    --data_source "$PHYSICS_DATA_DIR"
fi

case "$VARIANT" in
  grpo_fsdp)
    CONFIG_NAME="grpo_fsdp_sciknoweval_physics_olmo_trainer.yaml"
    DEFAULT_EXP_NAME="olmo3_7b_section3_physics_grpo_fsdp"
    ;;
  sdpo_fsdp)
    CONFIG_NAME="sdpo_fsdp_sciknoweval_physics_olmo_trainer.yaml"
    DEFAULT_EXP_NAME="olmo3_7b_section3_physics_sdpo_fsdp"
    ;;
  sdpo_megatron)
    CONFIG_NAME="sdpo_megatron_sciknoweval_physics_qwen_trainer.yaml"
    DEFAULT_EXP_NAME="qwen3_8b_section3_physics_sdpo_megatron"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT"
    echo "Expected one of: grpo_fsdp, sdpo_fsdp, sdpo_megatron"
    exit 1
    ;;
esac

EXP_NAME="${EXP_NAME:-$DEFAULT_EXP_NAME}"

if [[ -z "${VALIDATION_DATA_DIR:-}" ]]; then
  if [[ "$PHYSICS_DATA_DIR" == /hai/* ]]; then
    VALIDATION_DATA_DIR="$REMOTE_VALIDATION_ROOT/$EXP_NAME"
  else
    VALIDATION_DATA_DIR="$LOCAL_VALIDATION_ROOT/$EXP_NAME"
  fi
fi

mkdir -p "$VALIDATION_DATA_DIR"

export MODEL_PATH
export PHYSICS_TRAIN_FILE
export PHYSICS_VAL_FILE
export N_GPUS_PER_NODE
export NNODES
export TRAIN_TP_SIZE
export TRAIN_PP_SIZE
export ROLLOUT_TP_SIZE
export ROLLOUT_PP_SIZE
export VALIDATION_DATA_DIR

python3 -m verl.trainer.main_ppo \
  --config-name "$CONFIG_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  "$@"
