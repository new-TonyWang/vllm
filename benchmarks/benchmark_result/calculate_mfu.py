
import os
import csv

class MFUModelConfig:
    def __init__(self, hidden_size, intermediate_size,
                 num_attention_heads, num_key_value_heads, num_hidden_layers,
                 a100_theoretical_flops,vocab_size, num_gpus):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.a100_theoretical_flops = a100_theoretical_flops
        self.vocab_size = vocab_size
        self.num_gpus = num_gpus

class MFUCalculator():
    def __init__(self,model_config:MFUModelConfig):
        self.config = model_config

    def calculate_decode_mfu(self,batch_size,seq_length,sum_kv_seq_length,throughput_tokens_per_sec):
        # 从ModelConfig实例中提取参数
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        # 前馈网络的FLOPs
        flops_qkv = 2 * batch_size * seq_length * hidden_size * (hidden_size + 2 * hidden_size // num_attention_heads * num_key_value_heads) / 1e12
        
        flops_proj = 2 * batch_size * seq_length * hidden_size * hidden_size / 1e12
        
        flops_attention = 2 * 2 * seq_length *(sum_kv_seq_length + batch_size * seq_length) * hidden_size / 1e12
        
        flops_ffn = 2 * 3 * seq_length * batch_size * hidden_size * intermediate_size / 1e12
        # 每层的FLOPs
        flops_per_layer = flops_attention + flops_ffn + flops_qkv + flops_proj
        
        lm_head = 2 * batch_size * seq_length * hidden_size * config.vocab_size / 1e12

        # 模型的总FLOPs
        total_flops = flops_per_layer * self.config.num_hidden_layers + lm_head

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
        flops_proj = 2 * batch_size * seq_length * hidden_size * hidden_size / 1e12
        flops_ffn = 2 * 3 * seq_length * batch_size * hidden_size * intermediate_size / 1e12
        # 每层的FLOPs
        flops_per_layer = flops_attention + flops_ffn + flops_qkv + flops_proj
        
        # 模型的总FLOPs
        total_flops = flops_per_layer * self.config.num_hidden_layers

        # 每个token的FLOPs
        flops_per_token = total_flops / (seq_length * batch_size)

        # 每秒的FLOPs
        flops_per_second = throughput_tokens_per_sec * flops_per_token

        # MFU计算
        mfu = flops_per_second / (self.config.a100_theoretical_flops * self.config.num_gpus)

        return mfu, flops_per_second

def calculate_mfu(config):
    # 从ModelConfig实例中提取参数
    seq_length = config.seq_length
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads

    # 前馈网络的FLOPs
    # [m, k] * [k * n]
    flops_qkv = 2 * batch_size * seq_length * hidden_size * (
                hidden_size + 2 * hidden_size // num_attention_heads * num_key_value_heads) / 1e12
    flops_attention = 2 * 2 * batch_size * seq_length * seq_length * hidden_size / 1e12
    flops_proj = 2 * batch_size * seq_length * hidden_size * hidden_size / 1e12
    flops_ffn = 2 * 3 * seq_length * batch_size * hidden_size * intermediate_size / 1e12
    # 每层的FLOPs
    flops_per_layer = flops_attention + flops_ffn + flops_qkv + flops_proj

    lm_head = 2 * batch_size * seq_length * hidden_size * config.vocab_size / 1e12

    # 模型的总FLOPs
    total_flops = flops_per_layer * config.num_hidden_layers + lm_head

    # 每个token的FLOPs
    flops_per_token = total_flops / (seq_length * batch_size)

    # 每秒的FLOPs
    flops_per_second = config.throughput_tokens_per_sec * flops_per_token

    # MFU计算
    mfu = flops_per_second / (config.a100_theoretical_flops * config.num_gpus)

    return mfu, flops_per_second


# 创建ModelConfig实例
config = MFUModelConfig(
    hidden_size=8192,
    intermediate_size=29568,
    num_attention_heads=64,
    num_key_value_heads=8,
    num_hidden_layers=80,
    a100_theoretical_flops=312,
    vocab_size=152064,
    num_gpus=8
)

mfu_calculator = MFUCalculator(config)

# 输入和输出文件名
input_directory = "/nas/shared/GAIR/wty/workspace/vllm/benchmarks/benchmark_result/benchmark_test_mfu_tp8/mfu_summary"
output_directory = "/nas/shared/GAIR/wty/workspace/vllm/benchmarks/benchmark_result/benchmark_test_mfu_tp8/mfu_result"
for file_name in os.listdir(input_directory):
    if file_name.endswith('.csv'):
        input_file = os.path.join(input_directory, file_name)
        output_file = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}_with_mfu.csv")
        # input_file = "/home/q00606281/output.csv"
        # output_file = "/home/q00606281/output_with_mfu.csv"

        # 读取 CSV 文件并增加 "mfu" 列
        with (open(input_file, mode="r", newline="", encoding="utf-8") as infile,
              open(output_file, mode="a", newline="", encoding="utf-8") as outfile):
            # 创建 CSV 读取器和写入器
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames + ["mfu"] + ["flops_per_second"]  # 添加新的列名 "mfu"
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            # 写入新的表头
            writer.writeheader()

            # 遍历每一行，计算 "mfu" 并写入新的文件
            for row in reader:
                # 将 max_seq_len, bsz, 和 num_tokens_per_sec 转换为数值类型
                p_or_d = str(row["p_or_d"])
                seq_length = int(row["max_seq_len"])
                batch_size = int(row["batch_size"])
                throughput_tokens_per_sec = float(row["num_tokens_per_sec"])
                if p_or_d == "prefill":
                    # 调用函数计算 "mfu"
                    (row["mfu"],row["flops_per_second"]) = mfu_calculator.calculate_prefill_mfu(batch_size=batch_size,seq_length=seq_length,throughput_tokens_per_sec=throughput_tokens_per_sec)
                elif p_or_d == "decode":
                    sum_kv_seq_length = int(row["total_num_actual_tokens"])
                    (row["mfu"],row["flops_per_second"]) = mfu_calculator.calculate_decode_mfu(batch_size,seq_length,sum_kv_seq_length,throughput_tokens_per_sec)
                else:
                    raise RuntimeError(f"Task catogory must be prefill or decode, unsupported type detected: {p_or_d}")

                # 写入这行数据到新的文件
                writer.writerow(row)


# # 计算MFU和每秒FLOPs
# mfu, flops_per_second = calculate_mfu(config)

# # 输出结果
# print(f"MFU: {mfu:.4f}")
# print(f"FLOPs per second: {flops_per_second:.2e}")




