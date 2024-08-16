import os
import warnings
import redis
import openai
import json

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from redisvl.extensions.llmcache import SemanticCache
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Text, Tag
from redisvl.utils.vectorize import HFTextVectorizer

#GLOBAL VARIABLES
rds = None
llmcache = None
em = None



def get_connection(REDIS_URL):
    redis_connection = redis.Redis.from_url(REDIS_URL)
    return redis_connection

def load_folder(folder_path, index):
    docs = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    for doc in docs:
        if os.path.isdir(doc) == False:
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

def get_user_dept(user):
    global rds
    return rds.hget(f"user:{user}", "dept")


#GET SECUIRTY FILTER
def get_security_filter(user):
    user_dept = get_user_dept(user)

    if user_dept == None:
        fc = Tag("sector") == "_NOT_AUTHORIZED_"
    elif user_dept == "GLOBAL" :
        fc = Tag("sector") == []
    else:   
        fc = Tag("sector") == f"{user_dept}"

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
        filter_orig = query.get_filter()
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
                        return_fields=["content"],
                        num_results= 5
                    )

    # add security filter
    add_query_filter(vector_query, get_security_filter(user))

    # add data filter
    add_query_filter(vector_query, get_data_filter(query))
    
    results = index.query(vector_query)
    content = "\n".join([result["content"] for result in results])
    return content

#ANSWER QUERY - via RAG
def answer_question(index, query, user):

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
            cache_department = cached_result['metadata']['dept']
            #GLOBAL can see all cached responses, except negative response
            if get_user_dept(user) == "GLOBAL" and cached_result['response'] != get_negative_response():
                answer = f"$$$ I found a similar question in my cache $$$\n{cached_result['response']}"
                return answer
            elif cache_department.find(get_user_dept(user)) > -1:
                answer = f"$$$ I found a similar question in my cache $$$\n{cached_result['response']}"
                return answer

   
    context = do_vector_search(index, query, user)
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

    #Add to semantic cache, only cache when there is no contextual filter
    if get_data_filter(query) == None:
        dept = get_user_dept(user)
        llmcache.store(
                prompt= get_query_part(query),
                response= answer,
                metadata={"dept": dept}
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


    #SEMANTIC CACHE
    global llmcache
    llmcache = SemanticCache(
            name="llmcache",                    
            prefix="llmcache",                   
            redis_client=rds,
            ttl = 600,  
            distance_threshold=0.16
        )
    

    #LOAD METADATA
    doc_meta = load_dict_from_file("./doc-metadata.json")
    for key, value in doc_meta.items():
        for attr, attr_val in value.items():
            rds.hset("metadata:doc:" + key, attr, attr_val)


    #INITIALIZE SEARCH INDEX
    index = init_index(REDIS_URL, "./index-schema-def.yaml")
    print(f"[redis-vl-app] Index Exists: {index.exists()}")

    #EMBEDDING MODEL
    print(f"[redis-vl-app] Loading the embedding model")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global em
    em = HFTextVectorizer("sentence-transformers/all-MiniLM-L6-v2")

    prompt_load_data = input("Would you like to reload data [y/n]: ")

    if prompt_load_data == "y":
        #CREATE INDEX, DROP OLD DATA
        index.create(overwrite=True, drop=True)

        #LOAD DATA
        abs_path = input("Enter Absolute Path of Document or Folder : ")
        print("[redis-vl-app] Loading Documents")
        if os.path.isdir(abs_path) == True:
            load_folder(abs_path, index)
        else:
            load_file(abs_path, index)
        print("[redis-vl-app] Document Loading Complete")


    #LOAD THE USER PROFILE FOR DATA GOVERNANCE
    user = input("Enter your user profile: ")
    user_sector = rds.hget(f"user:{user}", "dept")
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
            #QUERY THE DATA
            #results = do_vector_search(index, question, user)
            answer = answer_question(index, question, user)
            print(answer)

    
    clear_cache = input("Clear the LLM Cache ? [y/n]")
    if clear_cache == "y":
        llmcache.clear()




if __name__ == "__main__":
    main()