import os
import logging
from typing import Dict, List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Class for processing documents from various sources"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def load_pdfs(self, category_path: Optional[str] = None) -> List[Document]:
        """
        Load PDFs from a specific category path or all categories if not specified

        Args:
            category_path: Optional subfolder path within data_dir

        Returns:
            List of documents with metadata
        """
        documents = []

        # Determine the root directory to scan
        root_dir = os.path.join(self.data_dir, category_path) if category_path else self.data_dir

        if not os.path.exists(root_dir):
            logger.error(f"Directory {root_dir} does not exist")
            return documents

        # Walk through the directory structure
        for dirpath, _, filenames in os.walk(root_dir):
            # Extract category from path
            rel_path = os.path.relpath(dirpath, self.data_dir)
            category = rel_path if rel_path != '.' else ''

            # Process PDF files
            for filename in filenames:
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        logger.info(f"Processing PDF: {file_path}")
                        loader = PyPDFLoader(file_path)
                        pdf_docs = loader.load()

                        # Add metadata to documents
                        for doc in pdf_docs:
                            doc.metadata.update({
                                'source': file_path,
                                'filename': filename,
                                'category': category
                            })
                            documents.append(doc)
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better embedding and retrieval

        Args:
            documents: List of documents to split

        Returns:
            List of split documents with preserved metadata
        """
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return documents

    def process_all_documents(self) -> Dict[str, List[Document]]:
        """
        Process all documents and organize them by category

        Returns:
            Dictionary with category as key and list of documents as value
        """
        all_documents = {}

        # List all directories in the data directory
        categories = [d for d in os.listdir(self.data_dir)
                     if os.path.isdir(os.path.join(self.data_dir, d))]

        # Process documents for each category
        for category in categories:
            docs = self.load_pdfs(category)
            split_docs = self.split_documents(docs)
            all_documents[category] = split_docs
            logger.info(f"Processed {len(split_docs)} chunks for category: {category}")

        return all_documents
