    export VLLM_RPC_GET_DATA_TIMEOUT_MS=1800000000
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    # export VLLM_TORCH_PROFILER_DIR="/data/tywang/workspace/llm_test/rank_0_3_profile"
    export PYTHONPATH=/data/tywang/workspace/vllm_site-packages:$PYTHONPATH  
    # . /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/public/env/llama/bin/activate
    CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m vllm.entrypoints.openai.api_server --model "/data/tywang/workspace/models/Abel-7B/qwen2.5-math-72b-sft-t2" \
    --trust-remote-code \
    --port 5600 \
    --swap-space 32 \
    --gpu-memory-utilization 0.95 \
    --dtype auto --tensor-parallel-size 4 \
    --enable-prefix-caching \
    --chat-template "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '' }}{% endif %}{{'' + message['content'] + '' + ''}}{% endfor %}{% if add_generation_prompt %}{{ '' }}{% endif %}" \
    --max-model-len 20480 > server_api_0.log 2>&1 