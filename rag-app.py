import os
import warnings
import redis


from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores.redis import Redis



from langchain_community.document_loaders import UnstructuredFileLoader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI,OpenAI, OpenAIEmbeddings

from redisvl.extensions.llmcache import SemanticCache


def get_connection(REDIS_URL):
    redis_connection = redis.Redis.from_url(REDIS_URL)
    return redis_connection

def create_index(REDIS_URL, index_file):
    index =  SearchIndex.from_yaml(index_file)
    index.connect(REDIS_URL)
    index.create(overwrite=True)
    return

def get_prompt():

    # Define the LLM Prompt
    prompt_template = """Answer the question using provided context. You are allowed to get creative.\n\n.

    This should be in the following format:

    Question: [question here]
    Answer: [answer here]

    Begin!

    Context:
    ---------
    {context}
    ---------
    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return prompt

def split_data(file_path):
 
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    print(f"Split Data into {len(chunks)} chunks")

    return chunks

def get_vectorstore(REDIS_URL, INDEX_NAME, embedding_model, chunks) -> Redis:
    try:
        vectorstore = Redis.from_existing_index(embedding=embedding_model, index_name=INDEX_NAME,redis_url=REDIS_URL)
        print(f"Loaded Existing Index")
    except Exception as e:
        # Load Redis with documents
        vectorstore = Redis.from_documents( documents=chunks, embedding=embedding_model, index_name=INDEX_NAME, redis_url=REDIS_URL)
        print(f"Created Index {INDEX_NAME}")

    return vectorstore



def main():

    warnings.filterwarnings("ignore")

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_URL = f"redis://default:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

    #CHECK Connection
    r = get_connection(REDIS_URL)
    print(f"Check Connection: {r.ping()}")


    #EMBEDINGS - OpenAI
    embedding_model = OpenAIEmbeddings()
    #embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #SPLIT DATA
    print("Chunking Data..")
    chunks = split_data("./documents/HR-Announcements.pdf")

    # VECTOR DB - REDIS
    redis_vdb = get_vectorstore(REDIS_URL, "idx:rag", embedding_model, chunks) 


    # LLM - GPT 3.5 TURBO
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

    #CREATE A CHAIN
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                            retriever=redis_vdb.as_retriever(search_type="similarity_distance_threshold",                
                                                                             search_kwargs={"distance_threshold":0.5},
                                                                             chain_type_kwargs={"prompt": get_prompt()},
                                                                             verbose=True))
    
    #SEMANTIC CACHE
    print("Creating a Semantic Cache")
    llmcache = SemanticCache(
        name="llmcache",                    
        prefix="llmcache",                   
        redis_url=REDIS_URL,  
        distance_threshold=0.3
    )
        
    while True:
        question = input("==============================================================\nAsk me any question ...\n")
        if question == "bye" or question == "quit":
            break
        else:
            #First check the LLM CACHE for a similar question
            if response := llmcache.check(prompt=question):
                print("$$$         Returning a Cached Answer         $$$\n")
                print(response[0]['response'])
            else:
                res = rag_chain(question)
                answer = res['result']
                llmcache.store(prompt=question, response=answer, metadata = {"developer" : "jay"})
                print(f"{answer}")


    #DELETE CACHE INDEX AND KEYS
    print("Deleting Index and Semantic Cache")
    redis_vdb.drop_index("idx:rag", delete_documents=True, redis_url=REDIS_URL)
    llmcache.clear()



if __name__ == "__main__":
    main()