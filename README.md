# Intelligent RAG System

This project implements an advanced Retrieval-Augmented Generation (RAG) system that intelligently selects the most relevant context based on user queries. The system is designed to handle multiple document categories and provide accurate responses by retrieving information from the most appropriate sources.

## Features

- **Intelligent Context Selection**: Automatically determines which document categories are relevant to a user's query
- **Multi-Model Support**: Uses multiple embedding and LLM options (Groq, OpenAI, Google Gemini, HuggingFace, Ollama)
- **Category-Based Document Organization**: Organizes documents by category for better retrieval
- **MLFlow Integration**: Tracks experiment runs and performance metrics
- **Streamlit UI**: User-friendly chat interface
- **FastAPI Backend**: REST API for programmatic access
- **Modular Architecture**: Easily extensible components

## Setup

1. **Clone the repository** and create a Python virtual environment (optional but recommended).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**:
   - Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

4. **Required API Keys**:
   - For Groq: Get an API key from [groq.com](https://groq.com)
   - For OpenAI: Get an API key from [openai.com](https://openai.com)
   - For Google Gemini: Get an API key from [ai.google.dev](https://ai.google.dev)
   - For HuggingFace: Get an API key from [huggingface.co](https://huggingface.co)
   - For Ollama: Install locally from [ollama.ai](https://ollama.ai)

## Usage

### Embedding Documents

To embed documents and create vector embeddings, use the `embed_documents.py` script:

```bash
python src2/embed_documents.py --provider ollama
```

Options:
- `--category`: Specify a category to process (default: all categories)
- `--clear`: Clear existing collections before embedding
- `--provider`: Choose embedding provider: "openai", "google", "huggingface", or "ollama" (default: "ollama")
- `--model`: Specify a model name (defaults based on provider)

### Starting the Streamlit UI

```bash
streamlit run src2/streamlit_app.py
```

### Starting the API Server

```bash
uvicorn src2.api:app --host 0.0.0.0 --port 8000
```

## Project Structure

- `src2/`: Main code directory
  - `config.py`: Configuration settings and environment variables
  - `document_processor.py`: Handles document loading and text splitting
  - `embeddings.py`: Manages embedding models (Ollama, OpenAI, Google, HuggingFace)
  - `vector_store.py`: Chromadb vector storage implementation
  - `rag_chain.py`: Core RAG implementation with intelligent routing
  - `embed_documents.py`: Script for processing and embedding documents
  - `api.py`: FastAPI implementation
  - `streamlit_app.py`: Streamlit UI
  - `data/`: Folder containing categorized documents
  - `vectordb/`: Storage for vector embeddings

## How It Works

1. Documents are organized into categories in the `data/` folder.
2. The system embeds documents using the chosen embedding model.
3. When a query is received, the system:
   - Analyzes the query to determine relevant document categories
   - Retrieves the most relevant documents from those categories
   - Uses the LLM to generate a response based on the retrieved context

## MLFlow Integration

The system uses MLFlow to track:
- Document embedding metrics
- Query processing details
- Response generation statistics

Access MLFlow UI by running:
```bash
mlflow ui
```
