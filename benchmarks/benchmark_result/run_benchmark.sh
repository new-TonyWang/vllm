# CUDA_VISIBLE_DEVICES=0,1,2,3 \
export PYTHONPATH=/data/tywang/workspace/vllm_site-packages:$PYTHONPATH  
#  --model "/data/tywang/workspace/models/Abel-7B/qwen2.5-math-72b-sft-t2" \
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /data/tywang/workspace/vllm/benchmarks/benchmark_throughput.py \
 --model "/data/tywang/workspace/models/Abel-7B/qwen2.5-math-72b-sft-t2" \
 --trust_remote_code \
 --enable-prefix-caching \
 --gpu-memory-utilization 0.95 \
 --dtype auto --tensor-parallel-size 4 \
 --device cuda  \
 --use-v2-block-manager \
 --num-prompts 100 \
 --dataset /data/tywang/workspace/vllm/benchmarks/sonnet.txt \
 --output-json /data/tywang/workspace/vllm/benchmarks/benchmark_result/output_json.json > benchmark.log 2>&1