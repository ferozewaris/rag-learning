import os
import logging
from typing import Dict, List, Optional, Any
import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.documents import Document
from embeddings import EmbeddingProvider
from vector_store import VectorStoreManager
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagChain:
    """Intelligent RAG Chain that selects the right collection based on question context"""

    def __init__(self, vector_store_manager: VectorStoreManager, embedding_provider: EmbeddingProvider):
        """
        Initialize the RAG Chain

        Args:
            vector_store_manager: Vector store manager instance
            embedding_provider: Embedding provider instance
        """
        self.vector_store_manager = vector_store_manager
        self.embedding_provider = embedding_provider
        self.llm = self._initialize_llm()
        self.collections = vector_store_manager.get_all_collections()

        # Initialize MLFlow tracking
        self._initialize_mlflow()

    def _initialize_llm(self) -> ChatGroq:
        """Initialize the Groq LLM"""
        try:
            return ChatGroq(
                groq_api_key=config.GROQ_API_KEY,
                model_name=config.GROQ_CHAT_MODEL
            )
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {str(e)}")
            raise

    def _initialize_mlflow(self) -> None:
        """Initialize MLFlow tracking"""
        try:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(config.EXPERIMENT_NAME)
        except Exception as e:
            logger.error(f"Error initializing MLFlow: {str(e)}")
            # Continue without MLFlow if it fails

    def _determine_relevant_categories(self, query: str) -> List[str]:
        """
        Determine which categories are relevant to the query

        Args:
            query: User query

        Returns:
            List of relevant category names
        """
        # Use the LLM to determine relevant categories
        prompt = ChatPromptTemplate.from_template(
            """You are an intelligent routing system. Your task is to determine which document categories are relevant to the user's query.
            
            Available categories:
            {categories}
            
            User query: {query}
            
            Return only the names of the relevant categories, one per line. If you're unsure, include all potentially relevant categories.
            """
        )

        chain = prompt | self.llm | StrOutputParser()

        with mlflow.start_run(nested=True):
            mlflow.log_param("query", query)
            mlflow.log_param("available_categories", list(self.collections.keys()))

            try:
                result = chain.invoke({
                    "categories": "\n".join(self.collections.keys()),
                    "query": query
                })

                # Parse the result to get category names
                relevant_categories = [
                    category.strip()
                    for category in result.split("\n")
                    if category.strip() in self.collections
                ]

                mlflow.log_param("selected_categories", relevant_categories)

                # If no categories match, use all categories
                if not relevant_categories:
                    relevant_categories = list(self.collections.keys())

                return relevant_categories
            except Exception as e:
                logger.error(f"Error determining relevant categories: {str(e)}")
                # Fallback to all categories
                return list(self.collections.keys())

    def get_multi_collection_retriever(self, categories: List[str], k: int = 3):
        """
        Create a retriever that searches across multiple collections

        Args:
            categories: List of category names to search
            k: Number of documents to retrieve per collection

        Returns:
            Function that takes a query and returns documents
        """
        def retriever_function(query: str) -> List[Document]:
            all_docs = []

            with mlflow.start_run(nested=True):
                mlflow.log_param("query", query)
                mlflow.log_param("search_categories", categories)

                for category in categories:
                    if category in self.collections:
                        try:
                            # Get documents from this collection
                            collection_docs = self.collections[category].similarity_search(
                                query=query,
                                k=k
                            )
                            all_docs.extend(collection_docs)

                            # Log the number of documents retrieved from each category
                            mlflow.log_metric(f"docs_from_{category}", len(collection_docs))
                        except Exception as e:
                            logger.error(f"Error retrieving from {category}: {str(e)}")

                # Log total documents retrieved
                mlflow.log_metric("total_docs_retrieved", len(all_docs))

                # Sort documents by relevance (using similarity scores if available)
                # For now, we'll just take the top k*len(categories) documents
                return all_docs[:k*len(categories)]

        return retriever_function

    def build_chain(self):
        """
        Build the intelligent RAG chain

        Returns:
            Callable chain that takes a query and returns a response
        """
        # Define the template with context
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering questions based on the retrieved documents.
            
            Retrieved documents:
            {context}
            
            Question: {question}
            
            Please provide a detailed answer based on the retrieved documents. If the documents don't contain relevant information, say so.
            """
        )

        def retrieve_and_generate(query: str) -> str:
            with mlflow.start_run():
                mlflow.log_param("original_query", query)

                # Step 1: Determine relevant categories
                relevant_categories = self._determine_relevant_categories(query)

                # Step 2: Get retriever for these categories
                retriever = self.get_multi_collection_retriever(relevant_categories)

                # Step 3: Retrieve documents
                documents = retriever(query)

                # Format documents for context
                context = "\n\n".join([f"Document from {doc.metadata.get('category', 'unknown')} - {doc.metadata.get('filename', 'unknown')}:\n{doc.page_content}" for doc in documents])

                # Step 4: Generate response using the LLM
                chain = prompt | self.llm | StrOutputParser()

                try:
                    response = chain.invoke({
                        "context": context,
                        "question": query
                    })

                    # Log the response
                    mlflow.log_param("response_length", len(response))

                    return response
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    return f"I'm sorry, I encountered an error while generating a response: {str(e)}"

        return retrieve_and_generate
