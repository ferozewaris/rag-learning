# RAG PDF QA App

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using LangChain and a remote Hugging Face LLM. The app runs locally via Streamlit and lets you query information from your own PDF files.

## Setup

1. **Clone the repository** and create a Python virtual environment (optional but recommended).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Get a Hugging Face API token**:
   - Create a free account at [huggingface.co](https://huggingface.co/).
   - Generate a token with access to the Inference API.
4. **Configure the token**:
   - Copy `.env` and add your token:
     ```bash
     cp .env .env.local
     echo "HUGGINGFACEHUB_API_TOKEN=your_actual_token" > .env.local
     ```
   - The application loads this variable at runtime.

## Running the App

1. Place the PDF files you want to query in a folder on your machine.
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Enter the path to your PDF folder in the web interface, then ask questions about the content.

The application uses the `mistralai/Mistral-7B-Instruct-v0.1` model via the Hugging Face Inference API and embeds your PDF text locally using `sentence-transformers/all-MiniLM-L6-v2` with FAISS for retrieval.

## Notes

- The app works fully offline except for calls to the Hugging Face Inference API.
- Errors such as missing PDFs or API issues are reported in the Streamlit interface.
