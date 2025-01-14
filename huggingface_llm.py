from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

load_dotenv()

class HuggingfaceLlm:
    def __init__(self):
        self.llm = None

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
        self.llm = ChatHuggingFace(llm=llm_endpoint, verbose=True)

    def invoke_llm(self, prompt):
        return self.llm.invoke(prompt)
