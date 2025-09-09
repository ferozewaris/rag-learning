import os
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from embeddings import EmbeddingProvider
from vector_store import VectorStoreManager
from rag_chain import RagChain
from document_processor import DocumentProcessor
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
# embedding_provider = EmbeddingProvider()
# vector_store_manager = VectorStoreManager(embedding_provider.get_embeddings())
# document_processor = DocumentProcessor(config.DATA_DIR)
# rag_chain = RagChain(vector_store_manager, embedding_provider)

# Initialize the RAG chain
# chain = rag_chain.build_chain()

# Create FastAPI app
app = FastAPI(
    title="Intelligent RAG API",
    description="API for querying the intelligent RAG system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

class IndexRequest(BaseModel):
    category: Optional[str] = None

class IndexResponse(BaseModel):
    message: str
    documents_indexed: int

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        response = chain(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# Index documents endpoint
@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    try:
        # Use the background task to index documents
        background_tasks.add_task(index_documents_task, request.category)

        return IndexResponse(
            message=f"Indexing documents from category {request.category if request.category else 'all'} started in the background",
            documents_indexed=0
        )
    except Exception as e:
        logger.error(f"Error starting indexing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting indexing: {str(e)}"
        )

# # Background task for indexing documents
# async def index_documents_task(category: Optional[str] = None):
#     try:
#         # Process documents
#         if category:
#             docs = document_processor.load_pdfs(category)
#             split_docs = document_processor.split_documents(docs)
#             vector_store_manager.add_documents(category, split_docs)
#             logger.info(f"Indexed {len(split_docs)} documents from category {category}")
#         else:
#             # Process all categories
#             all_documents = document_processor.process_all_documents()
#             total_docs = 0
#
#             for category, docs in all_documents.items():
#                 vector_store_manager.add_documents(category, docs)
#                 total_docs += len(docs)
#
#             logger.info(f"Indexed {total_docs} documents from all categories")
#     except Exception as e:
#         logger.error(f"Error indexing documents: {str(e)}")

# Upload document endpoint
# @app.post("/upload", response_model=IndexResponse)
# async def upload_document(
#     file: UploadFile = File(...),
#     category: str = Form(...)
# ):
#     try:
#         # Create category directory if it doesn't exist
#         category_dir = os.path.join(config.DATA_DIR, category)
#         os.makedirs(category_dir, exist_ok=True)
#
#         # Save the file
#         file_path = os.path.join(category_dir, file.filename)
#         with open(file_path, "wb") as f:
#             content = await file.read()
#             f.write(content)
#
#         # Process the document
#         loader = PyPDFLoader(file_path)
#         docs = loader.load()
#
#         # Add metadata
#         for doc in docs:
#             doc.metadata.update({
#                 'source': file_path,
#                 'filename': file.filename,
#                 'category': category
#             })
#
#         # Split documents
#         split_docs = document_processor.split_documents(docs)
#
#         # Add to vector store
#         vector_store_manager.add_documents(category, split_docs)
#
#         return IndexResponse(
#             message=f"Document {file.filename} uploaded and indexed in category {category}",
#             documents_indexed=len(split_docs)
#         )
#     except Exception as e:
#         logger.error(f"Error uploading document: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error uploading document: {str(e)}"
#         )

# Run the FastAPI app
def start_api():
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )

if __name__ == "__main__":
    start_api()
