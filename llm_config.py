import os
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.1") -> HuggingFaceHub:
    """Configure and return the remote Hugging Face LLM."""
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not set in environment")

    # Setup HuggingFaceHub LLM
    llm = HuggingFaceHub(
        repo_id=model_name,
        huggingfacehub_api_token=api_token,
        model_kwargs={"temperature": 0.0, "max_new_tokens": 512},
    )
    return llm
