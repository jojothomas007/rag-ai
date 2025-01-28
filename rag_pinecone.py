from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from huggingface_llm import HuggingfaceLlm
from pinecone import Pinecone, ServerlessSpec
import os

index_name = "langchain-doc-index2"
load_dotenv()

# Function to retrieve documents 
def retrieve_documents(query, top_k=5):
    pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    # Define the embedding model 
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    # query_embedding = embedding_model.embed_query(query) 
    results = vector_store.similarity_search(query) 
    return results 

# Function to generate a response using OpenAI's GPT-3 
def generate_response(user_prompt, system_prompt, context_documents): 
    context = " ".join([doc.page_content for doc in context_documents]) 
    prompt = [("system", "{0}. Context : {1}".format(system_prompt, context)),
            ("human", user_prompt),]
    llm = HuggingfaceLlm()
    llm.init_llm()
    response = llm.invoke_llm(prompt)
    return response

if __name__ == "__main__":
    user_prompt = "write scenario for Credit Scoring"
    system_prompt = "Genetrate 1 main Test scenario in Table format with Headers Step No, action, Expected resul, Actual Result, Precoditions"
    print("query - {0}".format(user_prompt))
    documents = retrieve_documents(user_prompt)
    print("Completed Retreival of relevant documents!")
    response = generate_response(user_prompt, system_prompt, documents)
    print(response)
    for text in response.content.split("\n"):
        print(text)
