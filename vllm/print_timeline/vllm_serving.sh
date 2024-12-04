export VLLM_TORCH_PROFILER_DIR="/data/tywang/workspace/llm_test/profile_out"
export PYTHONPATH=/data/tywang/workspace/vllm_site-packages:$PYTHONPATH  
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /data/ckpts/Baichuan2-13B-Base --host 127.0.0.1 --port 1025 --enable-prefix-caching --trust_remote_code \
--chat-template "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '' }}{% endif %}{{'' + message['content'] + '' + ''}}{% endfor %}{% if add_generation_prompt %}{{ '' }}{% endif %}" \
    