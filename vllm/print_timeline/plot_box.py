import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 假设数据存储在一个字符串中，每行是一个JSON对象

# 将字符串分割成单独的JSON对象
with open("/data/tywang/workspace/o1-journey-search-huawei/constructTree-mcts_20241128/examples/run_scripts_0.8_并行度100_phase1/timeline.txt","r") as f:
    data = f.readlines()
    json_objects = [json.loads(line) for line in data]

# 过滤出我们需要的数据
filtered_data = [
    (obj['args']['BatchSize'], obj['dur'])
    for obj in json_objects
    if 'BatchSize' in obj.get('args', {}) and ( 'Decode' in obj['name'])
]
# 创建DataFrame
df = pd.DataFrame(filtered_data, columns=['BatchSize', 'Duration(ms)'])
df['Duration(ms)'] = df['Duration(ms)']/1000
# 定义分组区间
bins = [i*10 for i in range(11)]  # 从0到100，每10为一组

# 使用cut函数将BatchSize划分到不同的区间
df['BatchGroup'] = pd.cut(df['BatchSize'], bins=bins)

# 绘制箱型图
plt.figure(figsize=(10, 6))
plt.ylim(0, 600)
sns.boxplot(x='BatchGroup', y='Duration(ms)', data=df)
plt.title('Decode time')
plt.xlabel('Batch Size Group')
plt.ylabel('Duration(ms)')
plt.show()
# 保存图表到本地
output_file = 'batch_size_duration_boxplot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Chart saved to {output_file}")