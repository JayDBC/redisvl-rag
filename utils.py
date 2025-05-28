import os
import warnings
import redis
import json

import time
from datetime import datetime

from transformers import pipeline
import torch

class RateLimiter:
    def __init__(self, redis_connection : redis.Redis, key_prefix="rag:limiter:", window=20):
        self.rds = redis_connection
        self.key_prefix = key_prefix
        self.window = window

    def is_allowed(self, user_key):
        allow_call = True

        key = f"{self.key_prefix}{user_key}"

        current_limit = self.rds.hincrby(key, "call_count", 1)
        self.rds.hexpire(key, self.window, "call_count", nx = True)

        user_limit = self.rds.hget(user_key, "call_limit")

        if current_limit > int(user_limit):
            allow_call = False

        return allow_call

class HFLocalLLM:
    def __init__(self, model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.pipe = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device_map="auto")
       
    def ask_llm(self, question):
        classmessages = [{
                    "role": "system",
                    "content": "You are a friendly chatbot who is meant to answer general knowledge questions, if you get asked a personal questions poitely refuse to answer"},
                    {"role": "user", 
                    "content": question}]
        prompt = self.pipe.tokenizer.apply_chat_template( classmessages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        reply = outputs[0]['generated_text']
        answer = reply[reply.find("<|assistant|>\n") + 14:]    
        return answer

def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def time_difference(ts1, ts2):
    # Convert Unix timestamps to datetime objects
    dt1 = datetime.fromtimestamp(ts1)
    dt2 = datetime.fromtimestamp(ts2)
    delta = abs(dt2 - dt1)
    minutes, seconds = divmod(delta.total_seconds(), 60) 
    return f"{int(minutes)}:{int(seconds)}"

def main():
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_URL = f"redis://default:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}?decode_responses=True"
    redis_connection = redis.Redis.from_url(REDIS_URL)

    rate_limiter = RateLimiter(redis_connection, "user:", 10)
    rate_limiter.allow("jay")

if __name__ == "__main__":
    main()
