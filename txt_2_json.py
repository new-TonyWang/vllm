import json
import time
timeline_list = []
timeline_list_json = []
data = None
marks = []
x_events = []
with open("/data/tywang/workspace/o1-journey-search-huawei/constructTree-mcts_20241128/examples/run_scripts_0.8_并行度100_phase1/timeline.txt","r")as f:
# with open("/data/tywang/workspace/llm_test/vllm_timeline/timeline_2024_12_03_08_14_42.txt","r")as f:
# with open("/data/tywang/workspace/llm_test/vllm_timeline/timeline_2024_12_03_08_14_42.json","r")as f:
    timeline_list = f.readlines()
    for i in timeline_list:
    #   print(i)
        item = json.loads(i)
        if item['ph'] == "M":
            marks.append(item)
        elif item['ph'] == 'X' and "Prefill"not in item['name'] and "Decode"not in item['name'] :
            x_events.append(item)
        else:
            if "Decode" in item['name']:
                item["cname"] = "rail_response"
            timeline_list_json.append(item)



# marks = [item for item in timeline_list_json if item['ph'] == 'M']
# x_events = [item for item in timeline_list_json if item['ph'] == 'X']

# 按照'ph':'X'的'ts'属性排序
x_events.sort(key=lambda x: x['ts'])

# 重新构建数据列表，保持原有的'M'与'X'配对关系
sorted_data = []
for event in x_events:
    # 找到与当前mark对应的X事件
    corresponding_x = next((x for x in marks if x['name'] == event['name']), None)
    if corresponding_x is not None:
        sorted_data.append(event)
        sorted_data.append(corresponding_x)
sorted_data.extend(timeline_list_json)

now = int(round(time.time()*1000))
now02 = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(now/1000))
output_json_file = f"/data/tywang/workspace/llm_test/vllm_timeline/timeline_{now02}.json"
print(output_json_file)
with open(output_json_file,"w")as f:
    json.dump(sorted_data,f,indent=4)