import os
import streamlit as st
import pandas as pd
import mlflow
from typing import Dict, List, Optional, Tuple
import config
from embeddings import EmbeddingProvider
from vector_store import VectorStoreManager
from rag_chain import RagChain
from document_processor import DocumentProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the page
st.set_page_config(
    page_title="Intelligent RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history and components
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    # Initialize components
    try:
        embedding_provider = EmbeddingProvider(
            provider=config.DEFAULT_EMBEDDING_PROVIDER,
            api_key=getattr(config, f"{config.DEFAULT_EMBEDDING_PROVIDER.upper()}_API_KEY", None),
            model_name=config.DEFAULT_EMBEDDING_MODEL
        )
        vector_store_manager = VectorStoreManager(
            embedding_provider.get_embeddings(),
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )
        document_processor = DocumentProcessor(config.DATA_DIR)

        # Initialize the RAG chain
        st.session_state.rag_chain = RagChain(vector_store_manager, embedding_provider)
        st.session_state.chain = st.session_state.rag_chain.build_chain()
        st.session_state.vector_store_manager = vector_store_manager
        st.session_state.document_processor = document_processor

        logger.info("Successfully initialized RAG components")
    except Exception as e:
        logger.error(f"Error initializing RAG components: {str(e)}")
        st.error(f"Error initializing the system: {str(e)}")

# Sidebar
with st.sidebar:
    st.title("Intelligent RAG System")
    st.markdown("This chatbot uses an intelligent RAG system to answer questions about different document categories.")

    # Display available categories
    st.subheader("Available Document Categories")

    # Get categories from data directory
    data_dir = config.DATA_DIR
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for category in categories:
        st.markdown(f"- {category}")

    # Add document indexing button
    st.subheader("Index Documents")

    if st.button("Index All Documents"):
        with st.spinner("Indexing documents... This may take a while."):
            try:
                # Process all documents using LangChain directly
                all_documents = st.session_state.document_processor.process_all_documents()
                total_docs = 0

                for category, docs in all_documents.items():
                    st.session_state.vector_store_manager.add_documents(category, docs)
                    total_docs += len(docs)

                st.success(f"Successfully indexed {total_docs} document chunks from all categories")
            except Exception as e:
                st.error(f"Error indexing documents: {str(e)}")

    # Add file uploader
    st.subheader("Upload New Document")

    upload_category = st.selectbox("Select Category", [""] + categories)
    new_category = st.text_input("Or Create New Category")

    selected_category = new_category if new_category else upload_category

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None and selected_category:
        if st.button("Upload and Index"):
            with st.spinner("Uploading and indexing document..."):
                try:
                    # Create category directory if it doesn't exist
                    category_dir = os.path.join(config.DATA_DIR, selected_category)
                    os.makedirs(category_dir, exist_ok=True)

                    # Save the file
                    file_path = os.path.join(category_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Process the document
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

                    # Add metadata
                    for doc in docs:
                        doc.metadata.update({
                            'source': file_path,
                            'filename': uploaded_file.name,
                            'category': selected_category
                        })

                    # Split documents
                    split_docs = st.session_state.document_processor.split_documents(docs)

                    # Add to vector store
                    st.session_state.vector_store_manager.add_documents(selected_category, split_docs)

                    st.success(f"Document {uploaded_file.name} uploaded and indexed in category {selected_category}")
                except Exception as e:
                    st.error(f"Error uploading document: {str(e)}")

    # Add information
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Ask questions about any of the document categories
    2. The system will intelligently retrieve the most relevant information
    3. You can upload new documents to expand the knowledge base
    """)

# Main chat interface
st.title("Intelligent Document Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_query = st.chat_input("Ask a question about any document category...")

# Process user input
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use LangChain directly instead of API call
                ai_response = st.session_state.chain(user_query)
                st.markdown(ai_response)

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)

                # Add error message to chat history
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add footer
st.markdown("---")
st.markdown("Intelligent RAG System powered by LangChain, ChromaDB, and MLFlow")
