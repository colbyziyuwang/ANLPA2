from llama_index.core import (
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.packs.sentence_window_retriever import SentenceWindowRetrieverPack as SentenceWindowRetriever
from llama_index.core.node_parser import SentenceWindowNodeParser

import torch

# ✅ Load the LLM Model using Llama2
llm = Ollama(
    model="llama3.2",
    context_window=4096,
    request_timeout=60.0,
    temperature=0.7
)

# ✅ Load the embedding model
embedding_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Configure Settings
Settings.llm = llm
Settings.embed_model = embedding_model

# ✅ Load documents
# file_path = "/content/drive/MyDrive/Advanced NLP/Assignments/data files/organized/10-K/0000001800/2001_0000912057-01-006039.txt"
file_path = "2001_0000912057-01-006039.txt"
docs = SimpleDirectoryReader(input_files=[file_path]).load_data()

# ✅ Create Node Parser with Sentence Window
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=1,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)

# ✅ Process nodes from documents
nodes = node_parser.get_nodes_from_documents(docs)

# ✅ Create Vector Store Index
index = VectorStoreIndex(nodes)

# ✅ Create Retriever
retriever = index.as_retriever(
    similarity_top_k=3
)

# ✅ Create Query Engine
query_engine = RetrieverQueryEngine(retriever=retriever)

# ✅ Function to run queries
def run_rag_query(query_text):
    response = query_engine.query(query_text)
    print("\n🔹 Query:", query_text)
    print("\n🔹 RAG Response:")
    print(response)
    return response

# ✅ Example usage
query = "What are the top 3-5 material risk factors highlighted in this 10-K?"
response = run_rag_query(query)

