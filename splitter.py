import os
import warnings
import json
import threading
import time

from redis import Redis
from datetime import datetime

from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
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

        pipe = self.rds.pipeline()

        loader = UnstructuredPDFLoader(file_path, mode="paged")
        docs  = loader.load()


        key = "chunk:doc:" + file_name

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)


        i = 1
        for doc in docs:
            # Splitting each page of document into chunks
            i = i + 1
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                pipe.xadd(key, {"file" : f"{file_name}" , "content" : chunk})

        pipe.execute()
        print(f"Loaded {file_name}")
        return  


    #CHUNK DATA
    def split_data(self, file_path):
        file_name = os.path.basename(file_path)
        loader = UnstructuredFileLoader(file_path, mode="single")
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
    print("Start Loading:")

    #["./documents/finance/aapl-10k-2023.pdf", "./documents/finance/amzn-10k-2023.pdf", "./documents/finance/jnj-10k-2023.pdf"]

    files = ["./documents/finance/aapl-10k-2023.pdf", "./documents/finance/amzn-10k-2023.pdf", "./documents/finance/jnj-10k-2023.pdf"]

    map = {}

    for file_path in files:
        file_name = os.path.basename(file_path)
        map[file_name] = threading.Thread(target=run_thread, args=(sp, file_path))
        map[file_name].start()

    for key, value in map.items():
        map[key].join()



    end_time = int(time.time())

    exec_time = time_difference(start_time, end_time)
    print(f"Execution Time: {exec_time}")


if __name__ == "__main__":
    main()