
from vllm.sequence import ExecuteModelRequest
class MFUModelConfig:
    def __init__(self, hidden_size, intermediate_size,
                 num_attention_heads, num_key_value_heads, num_hidden_layers,
                 a100_theoretical_flops, num_gpus):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.a100_theoretical_flops = a100_theoretical_flops
        self.num_gpus = num_gpus

class MFUCalculator():
    def __init__(self,model_config:MFUModelConfig):
        self.config = model_config
        
    # def query(self,batch_size,seq_length,throughput_tokens_per_sec):
    def calculate_mfu_adv(self,execute_model_req:ExecuteModelRequest,throughput_tokens_per_sec):
        # 从ModelConfig实例中提取参数
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        is_prefill = execute_model_req.seq_group_metadata_list[0].is_prompt
        num_past_kv_len = 0
        decode_batch = 0
        for metadata in execute_model_req.seq_group_metadata_list:
            for key in metadata.seq_data.keys():
                value = metadata.seq_data.get(key)
                all_prompt_length = value.get_prompt_len()
                num_new_token +=all_prompt_length
                num_past_kv_len += value.get_output_len()
                decode_batch += 1
        if is_prefill:
            num_past_kv_len = 0
            decode_batch=0
        else:
            num_new_token = 0
        
        # 前馈网络的FLOPs
        flops_qkv = 2 * (num_new_token+decode_batch) * hidden_size * (hidden_size + 2 * hidden_size // num_attention_heads * num_key_value_heads) / 1e12
        attention_seq_len = (num_new_token)
        flops_attention = 2 * 2 * batch_size * seq_length * seq_length * hidden_size / 1e12
        
        flops_ffn = 2 * 2 * seq_length * batch_size * hidden_size * intermediate_size / 1e12
        # 每层的FLOPs
        flops_per_layer = flops_attention + flops_ffn + flops_qkv

        # 模型的总FLOPs
        total_flops = flops_per_layer * self.config.num_hidden_layers

        # 每个token的FLOPs
        flops_per_token = total_flops / (seq_length * batch_size)

        # 每秒的FLOPs
        flops_per_second = throughput_tokens_per_sec * flops_per_token

        # MFU计算
        mfu = flops_per_second / (self.config.a100_theoretical_flops * self.config.num_gpus)

        return mfu, flops_per_second

    def calculate_decode_mfu(self,batch_size,seq_length,sum_kv_seq_length,throughput_tokens_per_sec):
        # 从ModelConfig实例中提取参数
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        # 前馈网络的FLOPs
        flops_qkv = 2 * batch_size * seq_length * hidden_size * (hidden_size + 2 * hidden_size // num_attention_heads * num_key_value_heads) / 1e12
        flops_attention = 2 * 2 * seq_length *(sum_kv_seq_length + batch_size * seq_length) * hidden_size / 1e12
        
        flops_ffn = 2 * 2 * seq_length * batch_size * hidden_size * intermediate_size / 1e12
        # 每层的FLOPs
        flops_per_layer = flops_attention + flops_ffn + flops_qkv

        # 模型的总FLOPs
        total_flops = flops_per_layer * self.config.num_hidden_layers

        # 每个token的FLOPs
        flops_per_token = total_flops / (seq_length * batch_size)

        # 每秒的FLOPs
        flops_per_second = throughput_tokens_per_sec * flops_per_token

        # MFU计算
        mfu = flops_per_second / (self.config.a100_theoretical_flops * self.config.num_gpus)

        return mfu, flops_per_second
    
    def calculate_prefill_mfu(self,batch_size,seq_length,throughput_tokens_per_sec):
        # 从ModelConfig实例中提取参数
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        # 前馈网络的FLOPs
        flops_qkv = 2 * batch_size * seq_length * hidden_size * (hidden_size + 2 * hidden_size // num_attention_heads * num_key_value_heads) / 1e12
        flops_attention = 2 * 2 * batch_size * seq_length * seq_length * hidden_size / 1e12
        
        flops_ffn = 2 * 2 * seq_length * batch_size * hidden_size * intermediate_size / 1e12
        # 每层的FLOPs
        flops_per_layer = flops_attention + flops_ffn + flops_qkv

        # 模型的总FLOPs
        total_flops = flops_per_layer * self.config.num_hidden_layers

        # 每个token的FLOPs
        flops_per_token = total_flops / (seq_length * batch_size)

        # 每秒的FLOPs
        flops_per_second = throughput_tokens_per_sec * flops_per_token

        # MFU计算
        mfu = flops_per_second / (self.config.a100_theoretical_flops * self.config.num_gpus)

        return mfu, flops_per_second

# 创建ModelConfig实例
# config = ModelConfig(
#     seq_length=4096,
#     batch_size=32,
#     hidden_size=8192,
#     intermediate_size=29568,
#     num_attention_heads=64,
#     num_key_value_heads=8,
#     num_hidden_layers=80,
#     throughput_tokens_per_sec=3000,
#     a100_theoretical_flops=312,
#     num_gpus=4
# )

# # 计算MFU和每秒FLOPs
# mfu, flops_per_second = calculate_mfu(config)

# # 输出结果
# print(f"MFU: {mfu:.4f}")
# print(f"FLOPs per second: {flops_per_second:.2e}")

