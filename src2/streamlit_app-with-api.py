import os
import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import config

# Configure the page
st.set_page_config(
    page_title="Intelligent RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# API endpoint
API_URL = f"http://localhost:{config.API_PORT}"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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
                response = requests.post(
                    f"{API_URL}/index",
                    json={}
                )
                if response.status_code == 200:
                    st.success("Indexing started in the background")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

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
                    files = {"file": uploaded_file}
                    data = {"category": selected_category}

                    response = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        data=data
                    )

                    if response.status_code == 200:
                        st.success(f"Document uploaded and indexed successfully")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

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
                # Call API
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": user_query}
                )

                if response.status_code == 200:
                    ai_response = response.json()["response"]
                    st.markdown(ai_response)

                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                else:
                    error_message = f"Error: {response.text}"
                    st.error(error_message)

                    # Add error message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)

                # Add error message to chat history
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add footer
st.markdown("---")
st.markdown("Intelligent RAG System powered by Groq, LangChain, ChromaDB, and MLFlow")
