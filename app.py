import os
import chainlit as cl
from dotenv import find_dotenv, dotenv_values
from operator import itemgetter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig

# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""

"""
We will load our environment variables here.
"""

keys = list(dotenv_values(find_dotenv('.env')).items())
HF_TOKEN = os.environ['HF_TOKEN'] = keys[0][1]
HF_EMBED_ENDPOINT = os.environ['HF_EMBED_ENDPOINT'] = keys[1][1]
HF_LLM_ENDPOINT = os.environ['HF_LLM_ENDPOINT'] = keys[2][1]

# ---- GLOBAL DECLARATIONS ---- #

# -- RETRIEVAL -- #
"""
1. Load Documents from Text File
2. Split Documents into Chunks
3. Load HuggingFace Embeddings (remember to use the URL we set above)
4. Index Files if they do not exist, otherwise load the vectorstore
"""
### 1. CREATE TEXT LOADER AND LOAD DOCUMENTS
### NOTE: PAY ATTENTION TO THE PATH THEY ARE IN. 
text_loader = TextLoader('./data/paul_graham_essays.txt')
documents = text_loader.load()

### 2. CREATE TEXT SPLITTER AND SPLIT DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

split_documents = text_splitter.split_documents(documents)

### 3. LOAD HUGGINGFACE EMBEDDINGS
hf_embeddings = HuggingFaceEndpointEmbeddings(model=HF_EMBED_ENDPOINT, 
                                              task="feature-extraction", #from the class instantiation example
                                              huggingfacehub_api_token=HF_TOKEN)

if os.path.exists("./data/vectorstore"):
    vectorstore = FAISS.load_local(
        "./data/vectorstore", 
        hf_embeddings, 
        allow_dangerous_deserialization=True # this is necessary to load the vectorstore from disk as it's stored as a `.pkl` file.
    )
    hf_retriever = vectorstore.as_retriever()
    print("Loaded Vectorstore")
else:
    print("Indexing Files")
    os.makedirs("./data/vectorstore", exist_ok=True)
    ### 4. INDEX FILES
    ### NOTE: REMEMBER TO BATCH THE DOCUMENTS WITH MAXIMUM BATCH SIZE = 32
    batch_size = 32
    vectorstore = FAISS.from_documents(split_documents[0:batch_size], hf_embeddings) #Initialize vector store with first bacth of 32
    for i in range(batch_size, len(split_documents), batch_size):
        if len(split_documents) - i < batch_size: #prevent a possible index out of range error
            vectorstore.add_documents(split_documents[i::])
        else:
            vectorstore.add_documents(split_documents[i:i+batch_size])
    vectorstore.save_local("./data/vectorstore")
        
hf_retriever = vectorstore.as_retriever()

# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""
### 1. DEFINE STRING TEMPLATE
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>

You are an assistant that exclusively answers questions based on the provided context. Any query that cannot be answered using the provided context should be politely refused, with an explanation as to why you could not answer.

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

"""

### 2. CREATE PROMPT TEMPLATE
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# -- GENERATION -- #
"""
1. Create a HuggingFaceEndpoint for the LLM
"""
### 1. CREATE HUGGINGFACE ENDPOINT FOR LLM
hf_llm = HuggingFaceEndpoint( 
    endpoint_url=HF_LLM_ENDPOINT,
    max_new_tokens=512, 
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_LLM_ENDPOINT
)

@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 

    In this case, we're overriding the 'Assistant' author to be 'Paul Graham Essay Bot'.
    """
    rename_dict = {
        "Assistant" : "Paul Graham Essay Bot"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    lcel_rag_chain = (
        {"context": itemgetter("query") | hf_retriever, "query": itemgetter("query")}
        | rag_prompt | hf_llm
    )

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    async for chunk in lcel_rag_chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()