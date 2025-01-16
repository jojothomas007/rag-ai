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
import reportlab 
import cairo
index_name = "langchain-doc-index"

def ingest_jira_docs():
    jira = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira)

    # Search for issues in a specific project 
    jql_query = "project = \'SCRUM\' AND assignee = currentUser()" 
    tool = None
    for the_tool in toolkit.tools:
        if the_tool.name == 'JQL Query':
            tool = the_tool

    # issues = toolkit.tools['JQL Query'](jql_query) 
    issues = tool(jql_query) 
    print(issues)
    context = " ".join([issue['key']['summary'] for issue in issues])

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs_split = text_splitter.split_text(context) 

    print(f"going to add {len(docs)} documents to pinecone")
    # add documents to vector db - if index is not already present, please create it using create_index()
    pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    vector_store.add_documents(documents=docs_split)


if __name__ == "__main__":
    print("Hello world")
    load_dotenv()
    ingest_jira_docs()
