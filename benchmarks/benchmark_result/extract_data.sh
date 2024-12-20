#!/bin/bash

# 输入文件名用于提取日志中的字段
INPUT_FILE="/data/tywang/workspace/o1-journey-search-huawei/constructTree-mcts_20241128/examples/run_scripts_0.8_优化序列数量/server_api_0.log"

# 提取数据
while read -r line; do
  # Total Decode tokens
  if [[ "$line" == *"Total Decode tokens"* ]]; then
    echo "$line" | awk -F "Total Decode tokens " '{print "Total Decode tokens: " $2}' | awk -F ", " '{print $1}'
  
  # Average decode time
  elif [[ "$line" == *"Average decode time"* ]]; then
    echo "$line" | awk -F "Average decode time:" '{print "Average decode time:" $2}' 
  
  # Decode speed
  elif [[ "$line" == *"Decode speed"* && "$line" != *"调度总时间"* ]]; then
    echo "$line" | awk -F "Decode speed:" '{print $2}'  | awk '{print "Decode speed:" $1}'
  
  # Total Prefill tokens
  elif [[ "$line" == *"Total Prefill tokens"* ]]; then
    echo "$line" | awk -F "Total Prefill tokens " '{print "Total Prefill tokens: " $2}' | awk -F ", " '{print $1}'
  
  # Average prefill time
  elif [[ "$line" == *"Average prefill time"* ]]; then
    echo "$line" | awk -F "Average prefill time:" '{print "Average prefill time:" $2}'
  
  # Prefill speed
  elif [[ "$line" == *"Prefill speed"* && "$line" != *"调度总时间"* ]]; then
    echo "$line" | awk -F "Prefill speed:" '{print $2}' | awk '{print "Prefill speed:" $1}'
  fi
done < "$INPUT_FILE"
