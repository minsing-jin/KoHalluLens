# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Set the environment variable for the API key
export BRAVE_API_KEY="your brave api key" 
export OPENAI_KEY="your openai key"

MODELS=(
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "meta-llama/Llama-3.1-405B-Instruct-FP8"    
)
for SEED in 0
do
    for MODEL in "${MODELS[@]}"
    do
        python -m tasks.refusal_test.round_robin_nonsense_name \
        --do_generate_prompt \
        --do_inference \
        --do_eval \
        --output_base_dir "output/refusal_test" \
        --generate_model $MODELS \
        --BUSINESS_N 500 \
        --EVENT_N 400 \
        --PRODUCT_N 100 \
        --BUSINESS_NAME_NUM 5 \
        --EVENT_NAME_NUM 4 \
        --PRODUCT_NAME_NUM 3 \
        --seed $SEED
    done
done