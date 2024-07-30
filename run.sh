#!/bin/bash

START=4096
END=256000
STEP=4096

MODEL_NAMES=(
  "tinyllama-110M"
  "LWM-Text-Chat-1M"
  "Yarn-Llama-2-7b-128k"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  python -u run_needle_in_haystack.py --s_len $START --e_len $END \
      --model_name $MODELS_DIR/${MODEL_NAME} \
      --attn_implementation flash_attention_2 \
      --step $STEP \
      --model_version ${MODEL_NAME}_${START}_${END}_${STEP}
done


