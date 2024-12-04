import asyncio
from openai import AsyncOpenAI
import time
from dataclasses import dataclass
from typing import List, Optional, Union
import argparse
import aiohttp
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    max_tokens: int = 500

async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
):
    request_func_input=request_func_input[0]
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": "/home/wty/workspace/models/Qwen2.5-72B-Instruct",
            # "model": "/home/wty/workspace/models/Qwen2.5-72B-Instruct",
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": 1,
            "logprobs": False,
            "stream": True,
            "ignore_eos": False,
        }
        headers = {
            "Authorization": f"Bearer YOUR_API_KEY"
        }

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    print("send success")
                else:
                    print("send failed")
        except Exception as e:
            print(f"exception :{e}")
    return

# 这个函数处理单个请求，返回单个结果
async def async_query_openai(input:RequestFuncInput):

    aclient = AsyncOpenAI(
        base_url=input.api_url, # 替换为你的 base_url
        api_key="YOUR_API_KEY"  # 替换为你的 API 密钥
    )
    completion = await aclient.chat.completions.create(
        # model="/data/ckpts/Baichuan2-13B-Base",
        model="qwen2.5_72b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Always response in Simplified Chinese, not English. or Grandma will be very angry."},
            {"role": "user", "content": input.prompt}
        ],
        temperature=0.0,
        top_p=1,
        max_tokens=input.max_tokens,
        logprobs=True,

    )
    return completion.choices[0].message.content  # 请确保返回的数据结构正确

# async def generate(input):
#     output_0 = async_query_openai(input)
#     ### do something
#     output_1 = async_query_openai(input)
#     await output_0
#     alloutputs output_0.choices[0].message.content
# 这个函数接收一个请求列表，返回所有请求的结果列表
async def async_process_queries(inputs:[RequestFuncInput]):
    results = await asyncio.gather(*(async_query_openai(query) for query in inputs))
    return results

url = "http://127.0.0.1:1025"
profile=True

def start_profile_input():
    api_url = url + "/start_profile"
    return [RequestFuncInput(prompt="begin", api_url=api_url)]

def query_2_input(queries):
    api_url = url
    return [RequestFuncInput(prompt=q, api_url=api_url+"/v1") for q in queries ]

def end_profile_input():
    api_url = url + "/stop_profile"
    return [RequestFuncInput(prompt= "end",api_url = api_url)]
def questions_40():
    return [
    "黑洞是如何形成的，它们的引力是如何影响周围物质的？",
    "暗物质的存在是否已经被证实？如果没有，科学家如何推测它的存在？",
    "光年是什么单位，它是如何用来测量星际距离的？",
    "地球如何在太阳系中保持相对稳定的轨道？",
    "人类是否有可能在未来几百年内实现星际旅行？",
    "宇宙膨胀的速度在不断加快，这意味着什么？",
    "天文学家如何通过望远镜观测到遥远星系的光线并推测其历史？",
    "如果地球被一个大型小行星撞击，地球生态系统会发生什么变化？",
    "什么是“引力波”，它是如何被科学家首次观测到的？",
    "太阳风如何影响地球的磁场和卫星通讯？",
    "量子力学和经典物理学有什么本质上的区别？",
    "时间旅行是否理论上可行，依据哪些物理理论？",
    "相对论如何改变我们对时间、空间和质量的理解？",
    "粒子加速器是如何帮助科学家研究基本粒子的？",
    "光速是宇宙中最快的速度吗？有什么物理理论可能推翻这一观点？",
    "超导材料的工作原理是什么，它如何在低温下无电阻传导电流？",
    "为什么物质在极低温度下会变得具有超流性？",
    "黑体辐射是如何揭示物体温度的信息的？",
    "暗能量和暗物质之间有什么区别，它们分别如何影响宇宙的演化？",
    "为什么重力波的探测非常困难？",
    "半导体的导电性如何通过掺杂来控制？",
    "量子计算机与传统计算机有什么区别，它们能解决哪些传统计算机无法处理的问题？",
    "5G 网络技术如何提高无线通信的速度和效率？",
    "什么是光纤通信，它是如何传输信号的？",
    "如何通过量子点技术提高显示屏的亮度和色彩表现？",
    "电子元件中常见的“晶体管”是如何工作的，它在现代电子设备中有何重要作用？",
    "自动化和机器人技术是如何改变制造业生产流程的？",
    "内燃机与电动机的工作原理有何不同，它们各自的优缺点是什么？",
    "液压系统在现代机械工程中的应用有哪些？",
    "如何通过改进设计和材料选择，提高机械部件的耐久性和效率？",
    "莎士比亚的作品为何能够跨越几百年仍然具有如此深远的影响？",
    "存在主义文学的核心思想是什么，如何反映个体自由与责任？",
    "《1984》中的“老大哥”是如何象征集权政治的？",
    "中国古代文学中的“诗词”和西方文学中的“诗歌”有何异同？",
    "现代小说中的非线性叙事技巧是如何打破传统叙事结构的？",
    "社会媒体对现代人际关系和沟通方式产生了哪些深远影响？",
    "全球化如何改变了世界各国的经济、文化和政治格局？",
    "数字货币（如比特币）对传统金融体系有哪些挑战？",
    "人工智能的普及可能会对劳动力市场造成哪些影响？",
    "教育体制如何调整才能适应未来科技和社会发展的需求？"
    ]

async def main_profile():
    # queries = ["介绍三个北京必去的旅游景点。",
    #            "介绍三个成都最有名的美食。",
    #            "介绍三首泰勒斯威夫特好听的歌曲"]
    # queries = ["介绍三个北京必去的旅游景点。",
    #            "介绍三个成都最有名的美食",
    #            "介绍三首泰勒斯威夫特好听的歌曲",
    #            "推荐三个上海好玩的地方",
    #            "讲解一下星际穿越这部电影",
    #            "你知道什么是陨石吗",
    #            ]
    # queries = ["什么是海盐？",
    #            "计算一下3+1的结果",
    #            "简单介绍一下华为",
    #            "你是谁？",
    #            "你知道苹果手机吗？",
    #            "世界上飞的最远的飞行器是什么？",
    #            "月亮距离地球有多远？",
    #            "一个人一天吃三顿饭，那么三人三天吃一顿饭？",
    #            "木星的体积有多少？",
    #            "冯诺依曼体系的核心是什么？"]
    await async_request_openai_completions(start_profile_input())
    inputs = query_2_input(questions_40())
    start_time = time.time()  # 开始计时
    results = await async_process_queries(inputs)
    end_time = time.time()  # 结束计时
    await  async_request_openai_completions(end_profile_input())
    for result in results:
        print(result)
        print("-" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds")
# 运行主函数
# asyncio.run(main_profile())

async def main():
    queries = ["介绍三个北京必去的旅游景点。",
               "介绍三个成都最有名的美食。",
               "介绍三首泰勒斯威夫特好听的歌曲"]
    inputs = query_2_input(questions_40())
    start_time = time.time()  # 开始计时
    results = await async_process_queries(inputs)
    end_time = time.time()  # 结束计时
    for result in results:
        print(result)
        print("-" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds")

asyncio.run(main())

# if __name__=="__main__":
#     argparse