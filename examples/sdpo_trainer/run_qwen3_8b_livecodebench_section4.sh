#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${VARIANT:-sdpo_fsdp}"
REMOTE_MODEL_PATH="/hai/zhoutong/section3_chemistry_assets/models/Qwen3-8B-Base"
REMOTE_LCB_DATA_DIR="/hai/zhoutong/section4_livecodebench_assets/data/lcb_v6"
LOCAL_LCB_DATA_DIR="/Users/zhoutong/code/SDPO/datasets/lcb_v6"
RAW_JSON_BASENAME="lcb_v6.json"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -f "$REMOTE_MODEL_PATH/config.json" ]]; then
    MODEL_PATH="$REMOTE_MODEL_PATH"
  else
    MODEL_PATH="Qwen/Qwen3-8B"
  fi
fi

if [[ -z "${LCB_DATA_DIR:-}" ]]; then
  if [[ -d "$REMOTE_LCB_DATA_DIR" || "$REMOTE_LCB_DATA_DIR" == /hai/* ]]; then
    LCB_DATA_DIR="$REMOTE_LCB_DATA_DIR"
  elif [[ -d "$LOCAL_LCB_DATA_DIR" ]]; then
    LCB_DATA_DIR="$LOCAL_LCB_DATA_DIR"
  else
    LCB_DATA_DIR="$REMOTE_LCB_DATA_DIR"
  fi
fi

LCB_TRAIN_FILE="${LCB_TRAIN_FILE:-$LCB_DATA_DIR/train.parquet}"
LCB_VAL_FILE="${LCB_VAL_FILE:-$LCB_DATA_DIR/test.parquet}"
RAW_JSON_PATH="${RAW_JSON_PATH:-$LCB_DATA_DIR/$RAW_JSON_BASENAME}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_PP_SIZE="${ROLLOUT_PP_SIZE:-1}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.55}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_REPROMPT_LEN="${MAX_REPROMPT_LEN:-6144}"
VALIDATION_ROOT="${VALIDATION_ROOT:-/hai/zhoutong/section4_livecodebench_assets/validation_generations}"
TEST_FREQ="${TEST_FREQ:-10}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-4}"
PRINT_VAL_GENERATIONS="${PRINT_VAL_GENERATIONS:-1}"

mkdir -p "$LCB_DATA_DIR"

if [[ ! -f "$LCB_DATA_DIR/train.json" || ! -f "$LCB_DATA_DIR/test.json" ]]; then
  if [[ ! -f "$RAW_JSON_PATH" ]]; then
    echo "Building LiveCodeBench v6 source JSON at $RAW_JSON_PATH ..."
    python3 "$REPO_ROOT/examples/data_preprocess/load_livecodebench_v6.py" \
      --output_path "$RAW_JSON_PATH"
  fi

  echo "Creating LiveCodeBench train/test JSON splits in $LCB_DATA_DIR ..."
  python3 "$REPO_ROOT/examples/data_preprocess/split_livecodebench_tests.py" \
    --json_path "$RAW_JSON_PATH" \
    --output_dir "$LCB_DATA_DIR"
fi

if [[ ! -f "$LCB_TRAIN_FILE" || ! -f "$LCB_VAL_FILE" ]]; then
  echo "Preprocessed parquet not found. Generating from $LCB_DATA_DIR ..."
  python3 "$REPO_ROOT/examples/data_preprocess/sdpo_generalization.py" \
    --data_source "$LCB_DATA_DIR"
fi

case "$VARIANT" in
  sdpo_fsdp|sdpo)
    CONFIG_NAME="section4_sdpo"
    DEFAULT_EXP_NAME="qwen3_8b_section4_livecodebench_sdpo_fsdp"
    ;;
  grpo_fsdp|grpo|baseline_grpo)
    CONFIG_NAME="section4_baseline_grpo"
    DEFAULT_EXP_NAME="qwen3_8b_section4_livecodebench_grpo_fsdp"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT"
    echo "Expected one of: sdpo_fsdp, sdpo, grpo_fsdp, grpo, baseline_grpo"
    exit 1
    ;;
esac

EXP_NAME="${EXP_NAME:-$DEFAULT_EXP_NAME}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER_OVERRIDE='trainer.logger=["console","wandb"]'
else
  LOGGER_OVERRIDE='trainer.logger=["console"]'
fi

VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$VALIDATION_ROOT/$EXP_NAME}"
mkdir -p "$VALIDATION_DATA_DIR"

export MODEL_PATH
export LCB_DATA_DIR
export N_GPUS_PER_NODE
export NNODES
export ROLLOUT_TP_SIZE
export ROLLOUT_PP_SIZE
export VLLM_GPU_MEM_UTIL
export MAX_PROMPT_LENGTH
export MAX_RESPONSE_LENGTH
export MAX_MODEL_LEN
export MAX_REPROMPT_LEN
export VERL_REPO_DIR="${VERL_REPO_DIR:-$REPO_ROOT}"

python3 -m verl.trainer.main_ppo \
  --config-name "$CONFIG_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  "$LOGGER_OVERRIDE" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
  trainer.print_val_generations="$PRINT_VAL_GENERATIONS" \
  trainer.validation_dump_generations=0 \
  "$@"
