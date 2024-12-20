import re
import matplotlib.pyplot as plt

# 定义日志文件路径
log_file1 = "/data/tywang/workspace/llm_test/base_line_data.txt"
log_file2 = "/data/tywang/workspace/llm_test/opt_seq_data.txt"

# 提取数据的函数
def extract_data(log_file):
    decode_tokens = []
    average_decode_time = []
    decode_speed = []

    with open(log_file, "r") as file:
        for line in file:
            # 匹配 Total Prefill tokens
            match_tokens = re.search(r"Total Prefill tokens: (\d+)", line)
            if match_tokens:
                decode_tokens.append(int(match_tokens.group(1)))

            # 匹配 Average decode time
            match_avg_time = re.search(r"Average prefill time:([\d.]+)", line)
            if match_avg_time:
                average_decode_time.append(float(match_avg_time.group(1)))

            # 匹配 Prefill speed（支持科学计数法）
            match_speed = re.search(r"Prefill speed:([\d.eE+-]+)", line)
            if match_speed:
                decode_speed.append(float(match_speed.group(1)))

    return decode_tokens, average_decode_time, decode_speed

# 提取两个日志文件的数据
decode_tokens1, average_decode_time1, decode_speed1 = extract_data(log_file1)
decode_tokens2, average_decode_time2, decode_speed2 = extract_data(log_file2)

decode_tokens1,decode_tokens2 = zip(*[x for x in zip(decode_tokens1, decode_tokens2)])
average_decode_time1,average_decode_time2 = zip(*[x for x in zip(average_decode_time1, average_decode_time2)])
decode_speed1,decode_speed2 = zip(*[x for x in zip(decode_speed1, decode_speed2)])


# 检查数据是否一致
if not (len(decode_tokens1) == len(average_decode_time1) == len(decode_speed1)):
    print(f"{log_file1} 数据不一致，请检查日志文件内容。")
    exit()
if not (len(decode_tokens2) == len(average_decode_time2) == len(decode_speed2)):
    print(f"{log_file2} 数据不一致，请检查日志文件内容。")
    exit()

# 创建子图
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# 子图 1: Total Prefill Tokens
axs[0].plot(range(len(decode_tokens1)), decode_tokens1, 
            label="baseline: Total Prefill Tokens", color="blue", markersize=4)
axs[0].plot(range(len(decode_tokens2)), decode_tokens2, 
            label="opt_max_seq_num: Total Prefill Tokens", color="red", markersize=4)
axs[0].set_title("Total Prefill Tokens")
axs[0].set_ylabel("Tokens")
axs[0].grid(True)
axs[0].legend()

# 子图 2: Average Prefill Time
axs[1].plot(range(len(average_decode_time1)), average_decode_time1, 
            label="baseline: Average Prefill Time", color="blue", markersize=4)
axs[1].plot(range(len(average_decode_time2)), average_decode_time2, 
            label="opt_max_seq_num: Average Prefill Time", color="red", markersize=4)
axs[1].set_title("Average Prefill Time")
axs[1].set_ylabel("Time (us)")
axs[1].grid(True)
axs[1].legend()

# 子图 3: Prefill Speed
axs[2].plot(range(len(decode_speed1)), decode_speed1, 
            label="baseline: Prefill Speed", color="blue",  markersize=4)
axs[2].plot(range(len(decode_speed2)), decode_speed2, 
            label="opt_max_seq_num: Prefill Speed", color="red",  markersize=4)
axs[2].set_title("Prefill Speed")
axs[2].set_xlabel("Index")
axs[2].set_ylabel("Speed")
axs[2].grid(True)
axs[2].legend()

# 图形样式
plt.tight_layout()

# 保存图形并显示
plt.savefig("compare_prefill_metrics.png")
plt.show()
