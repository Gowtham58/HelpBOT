
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings 
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv

PERSIST_DIR = "./db"

HF_TOKEN = ""

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


PERSIST_DIR = "./db"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")

#Create the new index
document = SimpleDirectoryReader("data").load_data() 
#parse docs into node
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(document)
#service context    
service_context = ServiceContext.from_defaults(llm=None, embed_model=embeddings, chunk_size=512)
storage_context = StorageContext.from_defaults()
index = VectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context)
index.storage_context.persist(persist_dir=PERSIST_DIR)   

print("Done Creating Database")