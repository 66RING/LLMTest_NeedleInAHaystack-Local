#!/bin/bash

python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001\
    --model_name $MODELS_DIR/tinyllama-110M \
    --attn_implementation flash_attention_2 \
    --step 100 \
    --model_version tinyllama_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}

