#!/bin/bash

export VLLM_RPC_GET_DATA_TIMEOUT_MS=1800000000
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export VLLM_TORCH_PROFILER_DIR="/data/tywang/workspace/llm_test/rank_0_3_profile"
export PYTHONPATH=/data/tywang/workspace/vllm_site-packages:$PYTHONPATH  

PYTHON_SCRIPT="/data/tywang/workspace/vllm/benchmarks/benchmark_throughput.py"

INPUT_LENS=(2048)
NUM_PROMPTS_LIST=(64)
OUTPUT_LENS=(8)
# INPUT_LENS=(2048)
# NUM_PROMPTS_LIST=(256)
# OUTPUT_LENS=(32)
output_dir="/data/tywang/workspace/vllm/benchmarks/benchmark_result/pp_benchmark_profile"
mkdir -p $output_dir
OUTPUT_CSV="$output_dir/benchmark_result.csv"
echo "input_len,output_len,num_prompts,elapsed_time,tokens_per_second" > "$OUTPUT_CSV"

# profile_cmd="nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --export sqlite --force-overwrite true -o analysis_test" 
export VLLM_TORCH_PROFILER_DIR=$output_dir
for input_len in "${INPUT_LENS[@]}"; do
    for num_prompts in "${NUM_PROMPTS_LIST[@]}"; do
        for output_len in "${OUTPUT_LENS[@]}"; do
            
            echo "正在运行: --input-len=$input_len, --num-prompts=$num_prompts ,--output-len=$output_len"
            log_name=input_lens_${input_len}_num_prompts_${num_prompts}_output_len_${output_len}.log
            echo "log 名称: ${log_name}"

            TEMP_JSON=$(mktemp)

            CUDA_VISIBLE_DEVICES=4,5,6,7 $profile_cmd python "$PYTHON_SCRIPT" \
                --model "/data/tywang/workspace/models/Abel-7B/qwen2.5-math-72b-sft-t2" \
                --use-v2-block-manager \
                --backend vllm \
                --input-len "$input_len" \
                --output-len $output_len \
                --dtype auto --pipeline-parallel-size 4 \
                --max-num-batched-tokens 65526 \
                --async-engine \
                --gpu-memory-utilization 0.95 \
                --max-num-seqs 1024 \
                --num-prompts "$num_prompts" \
                --profile \
                --output-json "$TEMP_JSON" &> "$output_dir/$log_name"

            # 使用 jq 提取所需的指标
            elapsed_time=$(jq '.elapsed_time' "$TEMP_JSON")
            tokens_per_second=$(jq '.tokens_per_second' "$TEMP_JSON")

            echo "$input_len,$output_len,$num_prompts,$elapsed_time,$tokens_per_second" >> "$OUTPUT_CSV"

            rm "$TEMP_JSON"

            echo "完成: elapsed_time=${elapsed_time}, tokens_per_second=${tokens_per_second}"
        done
    done
done

echo "BENCHMARK测试已完成。结果保存在 $OUTPUT_CSV"
