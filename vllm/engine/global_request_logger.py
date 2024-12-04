from vllm.logger import init_logger
import time
import threading
import jsonlines
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)
from dataclasses import dataclass, field, fields
import math
import json
from copy import deepcopy
BEGIN="BeginTime"

logging = init_logger(__name__)

@dataclass
class BatchInfo():
    batch:int = 0
    def __str__(self):
        return f"BatchSize:{self.batch}"
    def to_dict(self):
        return {"BatchSize":self.batch}

def create_event(name, ph, ts,dur, tid, pid=0, args=None):
    return {
        "name": name,
        "ph": ph,
        "ts": ts,
        "dur":dur,
        "tid": tid,
        "pid": pid,
        "args": args or {}
    }

class TimeLineTracing:
    def __init__(self, begin_time,system_begin_time):
        self.begin_time = begin_time
        self.system_begin_time = system_begin_time
        self.decode_begin_time = []
        self.decode_end_time = []
        self.prefill_time = -1
        self.prefill_end_time = -1
        self.finish_time = -1
        self.prefill_batch_info = None
        self.decode_batch_info = []

    def add_prefill_begin_time(self, time,bs):
        self.prefill_time = time
        self.prefill_batch_info = BatchInfo(bs)
    def add_prefill_end_time(self, time):
        self.prefill_end_time = time 

    def add_decode_end_time(self, time):
        self.decode_end_time.append(time )

    def add_decode_begin_time(self, time,bs):
        self.decode_begin_time.append(time)
        self.decode_batch_info.append(BatchInfo(bs))

    def add_swap_time(self, time):
        # 这个方法目前没有具体实现
        pass

    def add_finish_time(self, time):
        self.finish_time = time

    def export_timeline(self):
        # 可以根据需要扩展此方法来导出时间线数据到文件或其他格式
        print(str(self))

    def average_decode_time(self):
        accu = 0
        for i, (begin, end,info) in enumerate(zip(self.decode_begin_time, self.decode_end_time,self.decode_batch_info)):
           accu += end - begin
        return accu/len(self.decode_end_time)

    def to_chrome_tracing_events(self, request_id, pid=0):
        events = []

        # Finish 事件
        if self.finish_time != -1:
            events.append(create_event(request_id, "B", self.begin_time, request_id, pid, args={"value": 1}))
            events.append(create_event(request_id, "E", self.finish_time, request_id, pid, args={"value": 1}))
        # Prefill 开始事件
        if self.prefill_time != -1:
            events.append(create_event("Prefill", "B", self.prefill_time, request_id, pid, args=self.prefill_batch_info.to_dict()))

        # Prefill 结束事件
        if self.prefill_end_time != -1:
            events.append(create_event("Prefill", "E", self.prefill_end_time, request_id, pid))

        # Decode 事件
        for i, (begin, end, info) in enumerate(zip(self.decode_begin_time, self.decode_end_time, self.decode_batch_info)):
            events.append(create_event(f"Decode {i+1}", "B", begin, request_id, pid, args=info.to_dict()))
            events.append(create_event(f"Decode {i+1}", "E", end, request_id, pid))

        # Finish 事件
        if self.finish_time != -1:
            events.append(create_event("Finish", "I", self.finish_time, request_id, pid, args={"value": 1}))

        # 添加请求 ID 作为参数
        # for event in events:
        #     event["args"]["request_id"] = request_id

        return events

class TimeLine:
    def __init__(self, begin_time,system_begin_time,time_scale = 1000000):
        self.system_begin_time = system_begin_time*time_scale
        self.begin_time = begin_time*time_scale - self.system_begin_time
        self.time_scale = time_scale
        self.decode_begin_time = []
        self.decode_end_time = []
        self.prefill_time = -1
        self.prefill_end_time = -1
        self.finish_time = -1
        self.prefill_batch_info = None
        self.decode_batch_info = []
        self.request_token_num = 0

    def add_prefill_begin_time(self, time,bs):
        self.prefill_time = time*self.time_scale - self.system_begin_time
        self.prefill_batch_info = BatchInfo(bs)

    def add_prefill_end_time(self, time):
        self.prefill_end_time = time*self.time_scale - self.system_begin_time
        self.request_token_num+=1
    def add_decode_end_time(self, time):
        self.decode_end_time.append(time*self.time_scale - self.system_begin_time)
        self.request_token_num+=1
    def add_decode_begin_time(self, time,bs):
        self.decode_begin_time.append(time*self.time_scale - self.system_begin_time)
        self.decode_batch_info.append(BatchInfo(bs))

    def add_swap_time(self, time):
        # 这个方法目前没有具体实现
        pass

    def add_finish_time(self, time):
        self.finish_time = time*self.time_scale - self.system_begin_time

    def export_timeline(self):
        # 可以根据需要扩展此方法来导出时间线数据到文件或其他格式
        print(str(self))

    def get_request_token_num(self):
        return self.request_token_num

    def average_decode_time(self):
        accu = 0
        for i, (begin, end,info) in enumerate(zip(self.decode_begin_time, self.decode_end_time,self.decode_batch_info)):
           accu += end - begin
        return accu/len(self.decode_end_time)

    def to_string_short(self):
        timeline_str = f"Timeline:\n"
        timeline_str += f"  Begin Time: {self.begin_time}\n"
        if self.prefill_time != -1 and self.prefill_end_time != -1:
            timeline_str += f"  Prefill Start: {self.prefill_time} (from begin), Prefill End: {self.prefill_end_time} (from begin), Prefill Duration: {self.prefill_end_time-self.prefill_time} (from begin), {self.prefill_batch_info}\n"
        timeline_str += f"  Average Decode Time: {self.average_decode_time()} \n"
        bs = ""
        for info in self.decode_batch_info:
            bs += f"{info.batch},"
        timeline_str += f"  Decode Betches: [{bs}] \n"
        if self.finish_time != -1:
            timeline_str += f"  Finish Time: {self.finish_time} (from begin), Request Duration: {self.finish_time-self.begin_time}\n"
        return timeline_str

    def __str__(self):
        return self.to_string_short()
    
    def to_string_complete(self):
        timeline_str = f"Timeline:\n"
        timeline_str += f"  Begin Time: {self.begin_time}\n"
        if self.prefill_time != -1 and self.prefill_end_time != -1:
            timeline_str += f"  Prefill Start: {self.prefill_time} (from begin), Prefill End: {self.prefill_end_time} (from begin), Prefill Duration: {self.prefill_end_time-self.prefill_time} (from begin), {self.prefill_batch_info}\n"
            
        for i, (begin, end,info) in enumerate(zip(self.decode_begin_time, self.decode_end_time,self.decode_batch_info)):
            timeline_str += f"  Decode #{i+1} Start: {begin} (from begin), End: {end} (from begin), Duration: {end - begin}), {info} \n"

        if self.finish_time != -1:
            timeline_str += f"  Finish Time: {self.finish_time} (from begin), Request Duration: {self.finish_time-self.begin_time}\n"
        return timeline_str

    def to_chrome_tracing_events(self, request_id, pid=0):
        events = []

        # Finish 事件
        if self.finish_time != -1:
            events.append(create_event(request_id, "X",self.begin_time ,self.finish_time-self.begin_time, request_id, pid, args={"num tokens": self.get_request_token_num()}))
        # Prefill 开始事件
        if self.prefill_time != -1:
            events.append(create_event("Prefill", "X",self.prefill_time,self.prefill_end_time-self.prefill_time,request_id, pid, args=self.prefill_batch_info.to_dict()))

        # Decode 事件
        for i, (begin, end, info) in enumerate(zip(self.decode_begin_time, self.decode_end_time, self.decode_batch_info)):
            events.append(create_event(f"Decode {i+1}", "X",begin, end-begin, request_id, pid, args=info.to_dict()))

        return events


class GlobalRequestLogger():
    _instance_lock = threading.Lock()
    def __init__(self):
        # self.system_begin_time = time.time()
        self.reauest_id_2_timeline:Dict[str,TimeLine] = {}
        self.prompt_index = 0
        self.decde_index = 0
        self.token_generated = 0
        now = int(round(time.time()*1000))
        now02 = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(now/1000))
        self.timeline_export_file=f"/data/tywang/workspace/llm_test/vllm_timeline/timeline_{now02}.txt"
        logging.info(f"Saving timeline into, {self.timeline_export_file}")
        with open(self.timeline_export_file,"w") as f:
            f.write(" ")
        logging.info(f"GlobalRequestLogger init. ")
        self.begin = False

    def set_begin(self):
        self.begin = True
        self.system_begin_time = time.time()

    def is_begin(self):
        return self.begin
    
    def clear_timeline(self):
        self.reauest_id_2_timeline = {}

    def request_begin(self,request_id,time):
        if self.reauest_id_2_timeline.get(request_id):
            logging.warning(f"request id:{request_id} has already exist. ")

        self.reauest_id_2_timeline[request_id] = TimeLine(time,self.system_begin_time)
    def request_finish(self,request_id,time):
        if not self.reauest_id_2_timeline.get(request_id):
            logging.error(f"request id:{request_id} do not exist, but it has already finished ")
            return
        time_line = self.reauest_id_2_timeline[request_id]
        time_line.add_finish_time(time)

    def prompt_begin(self,request_id,time,bs):
        if not self.reauest_id_2_timeline.get(request_id):
            logging.error(f"request id:{request_id} do not exist, but it is prompting ")
            return
        time_line = self.reauest_id_2_timeline[request_id]
        time_line.add_prefill_begin_time(time,bs)
        self.prompt_index += 1

    def prompt_end(self,request_id,time):
        if not self.reauest_id_2_timeline.get(request_id):
            logging.error(f"request id:{request_id} do not exist, but prompting has been ended ")
            return
        time_line = self.reauest_id_2_timeline[request_id]
        time_line.add_prefill_end_time(time)

    def decode_end(self,request_id,time):
        if not self.reauest_id_2_timeline.get(request_id):
            # logging.error(f"request id:{request_id} do not exist, but decoding has been ended ")
            return
        time_line = self.reauest_id_2_timeline[request_id]
        time_line.add_decode_end_time(time)

    def decode_begin(self,request_id,time,bs):
        if not self.reauest_id_2_timeline.get(request_id):
            logging.error(f"request id:{request_id} do not exist, but it is decoding")
            return
        time_line = self.reauest_id_2_timeline[request_id]
        time_line.add_decode_begin_time(time,bs)
        self.decde_index += 1

    def print_timeline(self,request_id):
        if not self.reauest_id_2_timeline.get(request_id):
            logging.error(f"request id:{request_id} do not exist, but it is printing ")
            return
        time_line = self.reauest_id_2_timeline[request_id]
        self.token_generated += time_line.get_request_token_num()
        current_time = time.time()
        time_elapsed = current_time-self.system_begin_time
        log_str = f"{ str(time_line)} Total token generated {self.token_generated}, token/system_runing_time {self.token_generated/time_elapsed} token/second"
        logging.info(log_str)
        self.export_signle_trace(request_id,self.timeline_export_file)
        del self.reauest_id_2_timeline[request_id]

    def export_signle_trace(self,request_id,filename="timeline.txt"):
        timeline = self.reauest_id_2_timeline[request_id]
        all_events = []
        all_events.append({
            "name": request_id,
            "ph": "M",
            "pid": request_id,
            "tid": 0,
            "args": {"name": request_id}
        })
        all_events.extend(timeline.to_chrome_tracing_events(request_id))
        with open(filename,"a")as f:
            for i in all_events:
                f.write(json.dumps(i) + '\n')
        # del self.reauest_id_2_timeline[request_id]

    def export_to_chrome_tracing(self, filename="timeline.json"):
        all_events = []
        for request_id, timeline in self.reauest_id_2_timeline.items():
            # 添加 process 事件
            all_events.append({
                "name": request_id,
                "ph": "M",
                "pid": request_id,
                "tid": 0,
                "args": {"name": request_id}
            })
            all_events.extend(timeline.to_chrome_tracing_events(request_id))

        with open(filename, 'w') as f:
            json.dump(all_events, f, indent=4)

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(GlobalRequestLogger, "_instance"):
            with GlobalRequestLogger._instance_lock:
                if not hasattr(GlobalRequestLogger, "_instance"):
                    GlobalRequestLogger._instance = GlobalRequestLogger(*args, **kwargs)
        return GlobalRequestLogger._instance
    
    # @classmethod
    # def get_logger(cls):
    #     return GlobalRequestLogger()