from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

load_dotenv()

class HuggingfaceLlm:
    def __init__(self):
        self.llm_chat = None
        self.llm_endpoint = None

    def init_llm(self):
        llm_endpoint = HuggingFaceEndpoint(
        # repo_id="microsoft/phi-4",
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"],
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        )
        self.llm_endpoint = llm_endpoint
        self.llm_chat = ChatHuggingFace(llm=llm_endpoint, verbose=True)

    def invoke_llm(self, prompt):
        return self.llm_chat.invoke(prompt)
    
    def get_llm(self):
        return self.llm_endpoint
