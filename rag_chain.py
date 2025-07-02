import os
from typing import Optional
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever

from utils import load_pdfs_from_folder, split_documents
from llm_config import get_llm


class RAGPipeline:
    """Class to build and run a simple RAG pipeline."""

    def __init__(self, pdf_folder: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.pdf_folder = pdf_folder
        self.embedding_model = embedding_model
        self.vector_store: Optional[FAISS] = None
        self.retriever: Optional[BaseRetriever] = None
        self.qa_chain: Optional[RetrievalQA] = None

    def ingest_pdfs(self) -> None:
        """Load, split, embed PDFs and create FAISS vector store."""
        documents = load_pdfs_from_folder(self.pdf_folder)
        splits = split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = FAISS.from_documents(splits, embeddings)
        self.retriever = self.vector_store.as_retriever()

    def build_qa_chain(self) -> None:
        """Set up the RetrievalQA chain with the remote LLM."""
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Run ingest_pdfs() first.")

        llm = get_llm()
        self.qa_chain = RetrievalQA.from_chain_type(llm, retriever=self.retriever)

    def query(self, question: str) -> str:
        """Ask a question using the RAG pipeline."""
        if self.qa_chain is None:
            raise ValueError("QA chain not created. Run build_qa_chain() first.")
        result = self.qa_chain.run(question)
        return result
