import os
import warnings
import json
import threading
import time

from redis import Redis
from datetime import datetime

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from redisvl.extensions.llmcache import SemanticCache
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Text, Tag
from redisvl.utils.vectorize import HFTextVectorizer

class Splitter:
    def __init__(self, REDIS_URL):
        self.rds = Redis.from_url(REDIS_URL)
        print(f"Test Redis Connection: {self.rds.ping()}")

    def chunk_and_load(self, file_path) :
        file_name = os.path.basename(file_path)
        print(f"Starting {file_name}")
        chunks = self.split_data(file_path)
        pipe = self.rds.pipeline()
        i = 0
        key = "chunk:doc:" + file_name

        for chunk in chunks:
            pipe.xadd(key, {"file" : f"{file_name}" , "content" : chunk.page_content})

        pipe.execute()
        print(f"Loaded {file_name}")   


    #CHUNK DATA
    def split_data(self, file_path):
        file_name = os.path.basename(file_path)
        loader = UnstructuredFileLoader(file_path, mode="single", strategy="fast")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        chunks = loader.load_and_split(text_splitter)
        print(f"Chunking Complete {file_name}")
        return chunks


def run_thread(spl, file_path):
    spl.chunk_and_load(file_path)


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
    sp = Splitter(REDIS_URL)

    start_time = int(time.time())
    print("Start Chunking:")
    thread1 = threading.Thread(target=run_thread, args=(sp, "./documents/finance/aapl-10k-2023.pdf"))
    thread2 = threading.Thread(target=run_thread, args=(sp, "./documents/finance/amzn-10k-2023.pdf"))
    thread3 = threading.Thread(target=run_thread, args=(sp, "./documents/finance/jnj-10k-2023.pdf"))
    
    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()

    end_time = int(time.time())

    exec_time = time_difference(start_time, end_time)
    print(f"Execution Time: {exec_time}")


if __name__ == "__main__":
    main()