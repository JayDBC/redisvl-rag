import os
import warnings
import redis
import openai
import json

import time
from datetime import datetime

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from redisvl.extensions.llmcache import SemanticCache
from redisvl.extensions.router import Route, SemanticRouter

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Text, Tag
from redisvl.utils.vectorize import HFTextVectorizer

import utils
from utils import RateLimiter
from utils import HFLocalLLM

#GLOBAL VARIABLES
rds = None
llmcache = None
em = None
rate_limiter = None
local_llm = None



def get_connection(REDIS_URL):
    redis_connection = redis.Redis.from_url(REDIS_URL)
    return redis_connection

def load_folder(folder_path, index):
    docs = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    for doc in docs:
        if os.path.isdir(doc) == False and doc.find(".pdf") > -1:
            load_file(doc, index)

    return

def load_file(file_path, index):
    chunks = split_data(file_path)
    embeddings = create_embeddings(chunks)
    data = prepare_data(chunks, embeddings, file_path)
    load_redis(index, data)
    print(f"[redis-vl-app] Loaded Document: {file_path}")
    return

#CHUNK DATA
def split_data(file_path):
    loader = UnstructuredFileLoader(file_path, mode="single", strategy="fast")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    chunks = loader.load_and_split(text_splitter)
    #print(f"[redis-vl-app] Split {file_path} into {len(chunks)} chunks")
    return chunks

#CREATE EMBEDDINGS
def create_embeddings(chunks):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    embeddings = em.embed_many([chunk.page_content for chunk in chunks])
    #print(f"[redis-vl-app] Created Embeddings: {len(embeddings) == len(chunks)}")
    return embeddings

#PREPARE DATA
def prepare_data(chunks, embeddings, file_path):
    source_file = os.path.basename(file_path)
    file_name = os.path.splitext(source_file)[0]

    period = rds.hget(f"metadata:doc:{file_name}", "period")
    sector = rds.hget(f"metadata:doc:{file_name}", "sector")
    descriptor = rds.hget(f"metadata:doc:{file_name}", "descriptor")

    if descriptor == None:
        descriptor = ""

    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            "chunk_id" : f"{i}",
            "period" :  period,
            "sector" :  sector,
            "source" : source_file,
            "content" : "[" +  descriptor  + "] " + chunk.page_content ,
            "vector_embedding": embeddings[i]
        })

    return data


def init_index(REDIS_URL, index_file):
    index =  SearchIndex.from_yaml(index_file)
    index.connect(REDIS_URL)
    return index

def load_redis(index, data):
    index.load(data, preprocess=process_data)
    return index

def process_data(chunk):    
    return chunk

def get_user_sector(user):
    global rds
    return rds.hget(f"rag:users:{user}", "sector")


#GET SECUIRTY FILTER
def get_security_filter(user):
    user_sector = get_user_sector(user)
    user_sector = user_sector.replace(",","|")

    if user_sector == None:
        fc = Tag("sector") == "_NOT_AUTHORIZED_"
    elif user_sector == "GLOBAL" :
        fc = Tag("sector") == []
    else:   
        fc = Tag("sector") == f"{user_sector}"

    return fc

def get_data_filter(query):
    data_filter = None
    query = query.replace("must contain ", "_DELIM_").replace("filter by ", "_DELIM_")
    query_parts = query.split("_DELIM_")

    if len(query_parts) > 1:
        data_filter = Text("content") % query_parts[1].replace(", ", ",").replace(",", "|")

    return data_filter

def add_query_filter(query, filter):
    if filter != None:
        #filter_orig = query.get_filter()
        filter_orig = query.filter
        filter_new = filter_orig & filter
        query.set_filter(filter_new)
    return

def get_query_part(query):
    query = query.replace("must contain ", "_DELIM_").replace("filter by ", "_DELIM_")
    query_parts = query.split("_DELIM_")
    return query_parts[0]

#VECTOR SEARCH
def do_vector_search(index, query, user) -> str:

    query_vector = em.embed(get_query_part(query))

    vector_query = VectorQuery(
                        vector=query_vector,
                        vector_field_name="vector_embedding",
                        return_fields=["content","sector","period","source"],
                        num_results= 6
                    )

    # add security filter
    add_query_filter(vector_query, get_security_filter(user))

    # add data filter
    add_query_filter(vector_query, get_data_filter(query))
    
    # execute vector search
    results = index.query(vector_query)

    return results

def get_llm_context(results) -> str:
        content = "\n".join([result["content"] for result in results])
        return content


#ANSWER QUERY - via RAG
def answer_question(index, query, user, route_name):

    SYSTEM_PROMPT = """You are a helpful financial analyst assistant that has access
    to public financial 10k documents in order to answer users questions about company
    performance, ethics, characteristics, and core information.
    """

    answer = ""

    #Check LLM cache, do not check cache when explicit filters defined
    if get_data_filter(query) == None:
        cached_result_list = llmcache.check(prompt = get_query_part(query))
        if cached_result_list:
            cached_result = cached_result_list[0]
            cache_sector = cached_result['metadata']['sector']


            if get_user_sector(user) == "GLOBAL" and cached_result['response'] != get_negative_response():
                answer = f"[CACHED RESPONSE]\n{cached_result['response']}"
                return answer
            elif get_user_sector(user).find(cache_sector) > -1:
                answer = f"[CACHED RESPONSE]\n{cached_result['response']}"
                return answer


    #ROUTE TO LOCAL LLM
    if route_name != "finance":
        cached_result_list = llmcache.check(prompt = query)
        if cached_result_list:
            cached_result = cached_result_list[0]
            answer = f"[CACHED RESPONSE]\n{cached_result['response']}"
            return answer 
        elif route_name == "reject": 
            return "[Routing >> Guardrail] This question violates the terms of our user agreement & will not be answered."       
        else :
            print("[Route >> Local LLM] This is a generic question, I am passing this question to the local LLM")
            answer = local_llm.ask_llm(query)
            llmcache.store(
                prompt= query,
                response= answer,
                metadata={"sector": "general knowledge"}
            )
            return answer


    #RATE LIMIT CHECK
    allow_call = rate_limiter.is_allowed(f"rag:users:{user}")
    if allow_call == False:
        return "[LIMIT EXCEEDED]You have exceeded your API call limit, try after some time."

    #ROUTE TO RAG PIPELINE
    results = do_vector_search(index, query, user)
    context = get_llm_context(results)

    response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": get_prompt(get_query_part(query), context, get_negative_response())}
                    ],
                    temperature=0.1,
                    seed=42
                )
    # Response provided by LLM
    answer =  response.choices[0].message.content

    #Add to semantic cache, only cache when there is no contextual filter and positive answer
    if get_data_filter(query) == None and answer != get_negative_response():
        sector = results[0]['sector']
        llmcache.store(
                prompt= get_query_part(query),
                response= answer,
                metadata={"sector": sector}
            )

    return answer

def get_prompt(query: str, context: str, no_answer: str ) -> str:
    return f'''Use the provided context below derived from public financial
    documents to answer the user's question. 
    If you can't answer the user's question, based on the context; do not guess.
    Answer the question if and only if each piece of context contains the name of the coporation mentioned in the question, else respond with "{no_answer}"
    If there is no context at all, respond with "{no_answer}".

    User question:

    {query}

    Helpful context:

    {context}

    Answer:
    '''
    return

def get_negative_response():
    return "Sorry, I am unable to answer this question."

def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_dict_to_redis(file_path, key_prefix):
    data = load_dict_from_file(file_path)
    for key, value in data.items():
        for attr, attr_val in value.items():
            rds.hset(key_prefix + key, attr, attr_val)

def time_difference(ts1, ts2):
    # Convert Unix timestamps to datetime objects
    dt1 = datetime.fromtimestamp(ts1)
    dt2 = datetime.fromtimestamp(ts2)
    delta = abs(dt2 - dt1)
    minutes, seconds = divmod(delta.total_seconds(), 60) 
    return f"{int(minutes)}:{int(seconds)}"

#MAIN
def main():

    warnings.filterwarnings("ignore")

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_URL = f"redis://default:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}?decode_responses=True"

    #CHECK CONNECTION
    global rds
    rds = get_connection(REDIS_URL)
    print(f"[redis-vl-app] Test Redis Connection: {rds.ping()}")

    #EMBEDDING MODEL
    print(f"[redis-vl-app] Loading the embedding model")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global em
    em = HFTextVectorizer("sentence-transformers/all-MiniLM-L6-v2")    


    #SEMANTIC CACHE
    global llmcache
    llmcache = SemanticCache(
            name="llmcache",                    
            prefix="llmcache",                   
            redis_client=rds,
            ttl = 600,  
            distance_threshold=0.16
        )
    

    #LLM ROUTER
    llm_routes = utils.load_dict_from_file("./llm-router.json")

    llm_router = SemanticRouter(
        name="router",
        vectorizer = em,
        routes = llm_routes,
        redis_client = rds,
        overwrite=True
    )



    #RATE LIMITER
    global rate_limiter
    rate_limiter = RateLimiter(rds, "rag:limiter:", window=20)

    #LOCAL LLM
    global local_llm
    local_llm = HFLocalLLM()


    #LOAD USER PROFILES
    load_dict_to_redis("./user-profiles.json", "rag:users:")

    #INITIALIZE SEARCH INDEX
    index = init_index(REDIS_URL, "./index-schema-def.yaml")
    print(f"[redis-vl-app] Index Exists: {index.exists()}")



    prompt_load_data = input("Would you like to reload data [y/n]: ")

    if prompt_load_data == "y":
        #CREATE INDEX, DROP OLD DATA
        index.create(overwrite=True, drop=True)

        #LOAD DATA
        abs_path = input("Enter Absolute Path of Document or Folder : ")
        print("[redis-vl-app] Loading Documents")
        start_time = int(time.time())
        if os.path.isdir(abs_path) == True:
                #LOAD DOC METADATA
                load_dict_to_redis(os.path.join(abs_path, "doc-metadata.json"), "metadata:doc:")
                load_folder(abs_path, index)
        else:
            load_file(abs_path, index)

        end_time = int(time.time())
        exec_time = time_difference(start_time, end_time)
        print(f"[redis-vl-app] Document Loading Completed {exec_time}")


    #LOAD THE USER PROFILE FOR DATA GOVERNANCE
    user = input("Enter your user profile: ")
    user_sector = rds.hget(f"rag:users:{user}", "sector")
    if user_sector == None :
        print("Could not load your user profile, Please try Again !!")
        return
    else:
        print(f"\nHi {user}, you are permitted to view {user_sector}")



    while True:
        question = input("==============================================================\nAsk me any question ...\n")
        if question == "bye" or question == "quit":
            break
        else:   
            #CHECK THE ROUTE
            route_match = llm_router(question, distance_threshold= 2.0)
            route_name = route_match.name

            #QUERY THE DATA
            answer = answer_question(index, question, user, route_name)
            print(answer)

    
    clear_cache = input("Clear the LLM Cache ? [y/n]: ")
    if clear_cache == "y":
        llmcache.clear()




if __name__ == "__main__":
    main()