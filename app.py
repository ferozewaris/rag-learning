import streamlit as st
from rag_chain import RAGPipeline

st.title("RAG PDF QA")

# Input for folder containing PDFs
pdf_folder = st.text_input("Path to folder with PDFs")

if pdf_folder:
    if 'rag' not in st.session_state:
        try:
            rag = RAGPipeline(pdf_folder)
            rag.ingest_pdfs()
            rag.build_qa_chain()
            st.session_state['rag'] = rag
            st.success("PDFs loaded and embeddings created")
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")

    question = st.text_input("Ask a question about your PDFs")
    if question and 'rag' in st.session_state:
        try:
            answer = st.session_state['rag'].query(question)
            st.write(answer)
        except Exception as e:
            st.error(f"Error running query: {e}")
