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


def ingest_confluence_docs():
    # load documents from confluence
    loader = ConfluenceLoader(url=os.environ["CONFLUENCE_URL"], 
                              username=os.environ["JIRA_USERNAME"], 
                              api_key=os.environ["CONFLUENCE_API_TOKEN"],
                            #   token=os.environ["CONFLUENCE_API_TOKEN"],
                              space_key="SD",
                              limit=50,
                              )
    print("ReportLab version:", reportlab.Version) 
    print("Cairo version:", cairo.version)
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

def ingest_jira_docs():
    # Initialize Jira
    jira = JIRA(
        server=os.environ["JIRA_INSTANCE_URL"],
        basic_auth=(os.environ["JIRA_USERNAME"], os.environ["JIRA_API_TOKEN"])
    )
    # Define your JQL query to search for issues
    jql_query = "project = SCRUM AND assignee = currentUser()"

    # Search for issues
    issues = jira.search_issues(jql_query, maxResults=50)

    # Print issue details
    # for issue in issues:
    #     print(f"Key: {issue.key}, Summary: {issue.fields.summary}, description: {issue.fields.description}")

    context = " ".join([f"Key: {issue.key}, Summary: {issue.fields.summary}, description: {issue.fields.description}" for issue in issues])

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs_split = text_splitter.split_text(context) 

    print(f"going to add {len(issues)} issues to pinecone")
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
    # ingest_docs()
    # ingest_confluence_docs()
    ingest_jira_docs()
