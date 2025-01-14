from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os

index_name = "langchain-doc-index"

def ingest_docs():
    # load documents
    loader = DirectoryLoader("knowledge-base", 
                             glob=["**/*.txtw", "**/*.pdf"], 
                             silent_errors=True, 
                             show_progress=True)
    docs = loader.load()
    print(f"loaded {len(docs)} documents")
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs_split = text_splitter.split_documents(docs) 
    print(f"going to add {len(docs)} documents to pinecone")
    # add documents to vector db - if index is not already present, please create it using create_index()
    pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    vector_store.add_documents(documents=docs_split)


def create_index():
    pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
    pc.create_index(
        name=index_name,
        dimension=384, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )

if __name__ == "__main__":
    print("Hello world")
    load_dotenv()
    # create_index()
    ingest_docs()
