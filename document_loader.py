from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from jira import JIRA

index_name = "langchain-doc-index3"

def load_documents_from_drive():
    # load documents
    loader = DirectoryLoader("knowledge-base", 
                             glob=["**/*.txtw", "**/*.pdf"], 
                             silent_errors=True, 
                             show_progress=True)
    docs = loader.load()
    print(f"loaded {len(docs)} documents")
    return docs

def load_documents_from_confluence():
    # load documents from confluence
    loader = ConfluenceLoader(url=os.environ["CONFLUENCE_URL"], 
                              username=os.environ["JIRA_USERNAME"], 
                              api_key=os.environ["CONFLUENCE_API_TOKEN"],
                              space_key="SD",
                              limit=50,
                              )
    docs = loader.load()
    print(f"loaded {len(docs)} documents")
    return docs

def load_issue_details_from_jira():
    # Initialize Jira
    jira = JIRA(
        server=os.environ["JIRA_INSTANCE_URL"],
        basic_auth=(os.environ["JIRA_USERNAME"], os.environ["JIRA_API_TOKEN"])
    )
    # Define your JQL query to search for issues
    jql_query = "project = SCRUM AND assignee = currentUser()"

    # Search for issues
    issues = jira.search_issues(jql_query, maxResults=50)

    issue_details = " ".join([f"Key: {issue.key}, Summary: {issue.fields.summary}, description: {issue.fields.description}" for issue in issues])
    return issue_details


def create_index():
    pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
    if(not pc.has_index(index_name)):
        pc.create_index(
            name=index_name,
            dimension=384, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )

def split_documents(docs):
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs_split = text_splitter.split_documents(docs) 
    return docs_split

def split_text(text):
    # split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    text_split = text_splitter.split_text(text) 
    return text_split

# add documents to vector db - if index is not already present, creates it using create_index()
# docs_split: list of docs
def add_docs_to_vectordb(docs_split):
    create_index()
    pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    vector_store.add_documents(documents=docs_split)
    print(f"Added {len(docs_split)} document chunks to pinecone")

# add texts to vector db - if index is not already present, creates it using create_index()
# text_split: list of texts
def add_texts_to_vectordb(text_split):
    create_index()
    pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    vector_store.add_texts(texts=text_split)
    print(f"Added {len(text_split)} text chunks to pinecone")

def ingest_docs():
    docs = load_documents_from_drive()
    docs_split = split_documents(docs)
    add_docs_to_vectordb(docs_split)

def ingest_confluence_docs():
    docs = load_documents_from_confluence()
    docs_split = split_documents(docs)
    add_docs_to_vectordb(docs_split)

def ingest_jira_docs():
    issue_details = load_issue_details_from_jira()
    text_split = split_text(issue_details)
    add_texts_to_vectordb(text_split)

if __name__ == "__main__":
    print("Hello world")
    load_dotenv()
    ingest_docs()
    ingest_confluence_docs()
    ingest_jira_docs()
