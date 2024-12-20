python3 /data/tywang/workspace/vllm/benchmarks/benchmark_serving.py \
 --host localhost \
 --port 5600 \
 --dataset-name sonnet \
 --dataset-path /data/tywang/workspace/vllm/benchmarks/sonnet.txt \
 --num-prompts 100 \
 --model "/data/tywang/workspace/models/Abel-7B/qwen2.5-math-72b-sft-t2" \
 --endpoint "/v1/completions"