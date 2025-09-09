import os
import logging
from typing import Dict, List, Optional, Any
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Class for managing vector stores using ChromaDB"""

    def __init__(self, embeddings: Embeddings, persist_directory: Optional[str] = None):
        """
        Initialize the vector store manager

        Args:
            embeddings: Embeddings model
            persist_directory: Directory to persist vector store
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIRECTORY
        self.category_collections = {}

        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

    def create_collection(self, collection_name: str) -> VectorStore:
        """
        Create a new collection in ChromaDB

        Args:
            collection_name: Name of the collection

        Returns:
            Vector store for the collection
        """
        try:
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            collection = client.get_or_create_collection(name=collection_name)

            # Create vector store
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

            # Store in category collections
            self.category_collections[collection_name] = vector_store

            return vector_store
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {str(e)}")
            raise

    def add_documents(self, collection_name: str, documents: List[Document]) -> None:
        """
        Add documents to a collection

        Args:
            collection_name: Name of the collection
            documents: List of documents to add
        """
        try:
            # Get or create collection
            if collection_name not in self.category_collections:
                vector_store = self.create_collection(collection_name)
            else:
                vector_store = self.category_collections[collection_name]

            # Add documents
            vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to collection {collection_name}")
        except Exception as e:
            logger.error(f"Error adding documents to collection {collection_name}: {str(e)}")
            raise

    def get_collection(self, collection_name: str) -> Optional[VectorStore]:
        """
        Get a collection by name

        Args:
            collection_name: Name of the collection

        Returns:
            Vector store for the collection or None if not found
        """
        if collection_name in self.category_collections:
            return self.category_collections[collection_name]

        try:
            # Try to load collection
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection(name=collection_name)

            # Create vector store
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

            # Store in category collections
            self.category_collections[collection_name] = vector_store

            return vector_store
        except Exception as e:
            logger.error(f"Collection {collection_name} not found: {str(e)}")
            return None

    def get_all_collections(self) -> Dict[str, VectorStore]:
        """
        Get all collections

        Returns:
            Dictionary of collection names to vector stores
        """
        try:
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(path=self.persist_directory)

            # Get all collections
            collections = client.list_collections()

            # Load all collections
            for collection_info in collections:
                collection_name = collection_info.name
                if collection_name not in self.category_collections:
                    collection = client.get_collection(name=collection_name)
                    vector_store = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    self.category_collections[collection_name] = vector_store

            return self.category_collections
        except Exception as e:
            logger.error(f"Error getting all collections: {str(e)}")
            return self.category_collections
