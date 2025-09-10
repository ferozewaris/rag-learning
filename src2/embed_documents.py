import os
import argparse
import logging
from typing import Dict, List, Optional
import mlflow
from document_processor import DocumentProcessor
from embeddings import EmbeddingProvider
from vector_store import VectorStoreManager
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def embed_documents(category: Optional[str] = None,
                    clear_existing: bool = False,
                    provider: str = "ollama",
                    model_name: Optional[str] = None):
    """
    Process and embed documents from a specific category or all categories

    Args:
        category: Optional category to process. If None, process all categories
        clear_existing: Whether to clear existing collections before embedding
        provider: Embedding provider to use ('openai', 'google', 'huggingface', or 'ollama')
        model_name: Optional model name to use for embeddings
    """
    logger.info(f"Starting document embedding process for {'all categories' if not category else f'category: {category}'}")
    logger.info(f"Using {provider} as the embedding provider")

    try:
        # Initialize MLFlow tracking
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("document_embedding")

        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("category", category if category else "all")
            mlflow.log_param("clear_existing", clear_existing)
            mlflow.log_param("embedding_provider", provider)

            # Set the appropriate model name based on provider if not specified
            if not model_name:
                if provider == "openai":
                    model_name = config.OPENAI_EMBEDDING_MODEL
                elif provider == "google":
                    model_name = config.GOOGLE_EMBEDDING_MODEL
                elif provider == "huggingface":
                    model_name = config.HUGGINGFACE_EMBEDDING_MODEL
                elif provider == "ollama":
                    model_name = config.OLLAMA_EMBEDDING_MODEL
                else:
                    model_name = config.DEFAULT_EMBEDDING_MODEL

            mlflow.log_param("model_name", model_name)

            # Initialize components
            document_processor = DocumentProcessor(config.DATA_DIR)

            # Initialize the embedding provider with the specified provider
            if provider == "openai":
                embedding_provider = EmbeddingProvider(
                    provider="openai",
                    model_name=model_name,
                    api_key=config.OPENAI_API_KEY
                )
            elif provider == "google":
                embedding_provider = EmbeddingProvider(
                    provider="google",
                    model_name=model_name,
                    api_key=config.GOOGLE_API_KEY
                )
            elif provider == "huggingface":
                embedding_provider = EmbeddingProvider(
                    provider="huggingface",
                    model_name=model_name,
                    api_key=config.HUGGINGFACE_API_KEY
                )
            else:  # Default to ollama
                embedding_provider = EmbeddingProvider(
                    provider="ollama",
                    model_name=model_name,
                    ollama_base_url=config.OLLAMA_BASE_URL
                )

            vector_store_manager = VectorStoreManager(
                embedding_provider.get_embeddings(),
                persist_directory=config.CHROMA_PERSIST_DIRECTORY
            )

            # Process documents
            if category:
                # Process specific category
                logger.info(f"Processing documents from category: {category}")
                docs = document_processor.load_pdfs(category)
                logger.info(f"Loaded {len(docs)} documents from category: {category}")

                split_docs = document_processor.split_documents(docs)
                logger.info(f"Split into {len(split_docs)} chunks for category: {category}")

                # Log metrics
                mlflow.log_metric(f"original_docs_{category}", len(docs))
                mlflow.log_metric(f"split_docs_{category}", len(split_docs))

                # Add to vector store
                if clear_existing:
                    # Create a new collection (will overwrite if exists)
                    vector_store = vector_store_manager.create_collection(category)
                    vector_store.add_documents(split_docs)
                else:
                    # Add to existing collection
                    vector_store_manager.add_documents(category, split_docs)

                logger.info(f"Embedded {len(split_docs)} document chunks for category: {category}")

            else:
                # Process all categories
                logger.info("Processing documents from all categories")
                all_documents = document_processor.process_all_documents()

                total_original = 0
                total_split = 0

                for cat, docs in all_documents.items():
                    # Log metrics
                    mlflow.log_metric(f"split_docs_{cat}", len(docs))
                    total_split += len(docs)

                    # Add to vector store
                    if clear_existing:
                        # Create a new collection (will overwrite if exists)
                        vector_store = vector_store_manager.create_collection(cat)
                        vector_store.add_documents(docs)
                    else:
                        # Add to existing collection
                        vector_store_manager.add_documents(cat, docs)

                    logger.info(f"Embedded {len(docs)} document chunks for category: {cat}")

                # Log total metrics
                mlflow.log_metric("total_split_docs", total_split)

                logger.info(f"Embedded a total of {total_split} document chunks across all categories")

            logger.info("Document embedding process completed successfully")

    except Exception as e:
        logger.error(f"Error during document embedding process: {str(e)}")
        raise

def main():
    """Main entry point for the document embedding process"""
    parser = argparse.ArgumentParser(description="Embed documents for the RAG system")
    parser.add_argument(
        "--category",
        type=str,
        help="Specific category to process. If not provided, process all categories"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collections before embedding"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "google", "huggingface", "ollama"],
        default="ollama",
        help="Embedding provider to use (openai, google, huggingface, or ollama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use for embeddings (defaults based on provider)"
    )

    args = parser.parse_args()

    print("Arguments:")
    print(args.category, args.clear, args.provider, args.model)

    embed_documents(args.category, args.clear, args.provider, args.model)

if __name__ == "__main__":
    main()
