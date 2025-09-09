import os
import logging
from typing import Dict, List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """Class for providing embeddings using various providers"""

    def __init__(self,
                 provider: str = "openai",
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 ollama_base_url: Optional[str] = None):
        """
        Initialize the embedding provider with the selected provider

        Args:
            provider: The embedding provider to use ("openai", "google", "huggingface", or "ollama")
            model_name: Model name for embeddings
            api_key: API key for the provider (OpenAI, Google, HuggingFace)
            ollama_base_url: Base URL for Ollama API (only used if provider is "ollama")
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.ollama_base_url = ollama_base_url or config.OLLAMA_BASE_URL

        # Set default model name based on provider if not specified
        if not self.model_name:
            if self.provider == "openai":
                self.model_name = config.OPENAI_EMBEDDING_MODEL
            elif self.provider == "google":
                self.model_name = config.GOOGLE_EMBEDDING_MODEL
            elif self.provider == "huggingface":
                self.model_name = config.HUGGINGFACE_EMBEDDING_MODEL
            elif self.provider == "ollama":
                self.model_name = config.OLLAMA_EMBEDDING_MODEL
            else:
                self.model_name = config.DEFAULT_EMBEDDING_MODEL

        # Set API key based on provider if not specified
        if not self.api_key:
            if self.provider == "openai":
                self.api_key = config.OPENAI_API_KEY
            elif self.provider == "google":
                self.api_key = config.GOOGLE_API_KEY
            elif self.provider == "huggingface":
                self.api_key = config.HUGGINGFACE_API_KEY

        # Validate API key for providers that require it
        if self.provider in ["openai", "google"] and not self.api_key:
            raise ValueError(f"{self.provider.capitalize()} API key is required. Please set {self.provider.upper()}_API_KEY in .env file.")

        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self) -> Embeddings:
        """Initialize the embeddings model based on provider"""
        try:
            if self.provider == "openai":
                logger.info(f"Initializing OpenAI embeddings with model: {self.model_name}")
                return OpenAIEmbeddings(
                    openai_api_key=self.api_key,
                    model=self.model_name
                )
            elif self.provider == "google":
                logger.info(f"Initializing Google Gemini embeddings with model: {self.model_name}")
                return GoogleGenerativeAIEmbeddings(
                    google_api_key=self.api_key,
                    model=self.model_name
                )
            elif self.provider == "huggingface":
                logger.info(f"Initializing HuggingFace embeddings with model: {self.model_name}")
                return HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            elif self.provider == "ollama":
                logger.info(f"Initializing Ollama embeddings with model: {self.model_name}")
                return OllamaEmbeddings(
                    base_url=self.ollama_base_url,
                    model=self.model_name
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}. Use 'openai', 'google', 'huggingface', or 'ollama'.")
        except Exception as e:
            logger.error(f"Error initializing {self.provider} embeddings: {str(e)}")
            raise

    def get_embeddings(self) -> Embeddings:
        """Get the embeddings model"""
        return self.embeddings
