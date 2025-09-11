import os
import streamlit as st
import pandas as pd
import mlflow
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import config
from embeddings import EmbeddingProvider
from vector_store import VectorStoreManager
from rag_chain import RagChain
from document_processor import DocumentProcessor
from agent_router import AgentRouter
from duckdb_agent import DuckDBAgent
from logger import Logger

# Configure logging
logger = Logger(level="INFO", logger_name="STREAMLIT_APP")

# Configure the page
st.set_page_config(
    page_title="Multi-Agent Intelligent Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history and components
if "messages" not in st.session_state:
    st.session_state.messages = []

if "visualize_data" not in st.session_state:
    st.session_state.visualize_data = {}

if "rag_chain" not in st.session_state:
    # Initialize components
    try:
        # Initialize embedding provider
        embedding_provider = EmbeddingProvider(
            provider=config.DEFAULT_EMBEDDING_PROVIDER,
            api_key=getattr(config, f"{config.DEFAULT_EMBEDDING_PROVIDER.upper()}_API_KEY", None),
            model_name=config.DEFAULT_EMBEDDING_MODEL
        )

        # Initialize vector store manager
        vector_store_manager = VectorStoreManager(
            embedding_provider.get_embeddings(),
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )

        # Initialize document processor
        document_processor = DocumentProcessor(config.DATA_DIR)

        # Initialize the RAG chain
        st.session_state.rag_chain = RagChain(vector_store_manager, embedding_provider)
        st.session_state.chain = st.session_state.rag_chain.build_chain()
        st.session_state.vector_store_manager = vector_store_manager
        st.session_state.document_processor = document_processor

        # Initialize agent router
        st.session_state.agent_router = AgentRouter()

        # Initialize DuckDB agent
        st.session_state.duckdb_agent = DuckDBAgent(config.DUCKDB_PATH)

        # Create sample data for DuckDB if it doesn't exist
        # st.session_state.duckdb_agent.create_sample_data()

        logger.info("Successfully initialized all components")
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing the system: {str(e)}")

# Sidebar
with st.sidebar:
    st.title("Multi-Agent Intelligent Assistant")
    st.markdown("This chatbot routes questions to the appropriate agent based on the query type.")

    # Create tabs for different sidebar sections
    tab1, tab2, tab3 = st.tabs(["Document Data", "Structured Data", "Help"])

    with tab1:
        st.subheader("Document Categories")
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
                    # Process all documents using LangChain
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

    with tab2:
        st.subheader("Database Tables")
        # Display available tables
        tables = st.session_state.duckdb_agent.get_available_tables()
        for table in tables:
            st.markdown(f"- {table}")

        # Show sample data option
        if st.button("View Sample Data"):
            for table in tables:
                with st.expander(f"Table: {table}"):
                    try:
                        result = st.session_state.duckdb_agent.conn.execute(f"SELECT * FROM {table} LIMIT 5").fetchdf()
                        st.dataframe(result)
                    except Exception as e:
                        st.error(f"Error loading sample data: {str(e)}")

        # Option to reset sample data
        # if st.button("Reset Sample Data"):
        #     try:
        #         st.session_state.duckdb_agent.create_sample_data()
        #         st.success("Sample data has been reset successfully")
        #     except Exception as e:
        #         st.error(f"Error resetting sample data: {str(e)}")

    with tab3:
        st.subheader("How to use")
        st.markdown("""
        ### For Document Queries:
        - Ask questions about any document in the system
        - Upload new documents to expand the knowledge base
        
        ### For Data Analysis:
        - Ask questions about sales, products, or customers
        - Example: "Show me total sales by product category"
        - Example: "What's the average sales price by country?"
        
        The system will automatically route your question to the right agent!
        """)

# Function to generate appropriate visualization for data
def generate_visualization(df, query):
    """Generate an appropriate visualization based on the data and query"""
    try:
        logger.info("generating visualization")
        if len(df) == 0:
            return None

        # Determine chart type based on data shape and query
        chart_type = "bar"  # Default

        # Check if query contains keywords suggesting time series
        if any(word in query.lower() for word in ["time", "date", "trend", "over time", "monthly", "daily", "yearly"]):
            if any(col.lower() in ["date", "time", "year", "month", "day"] for col in df.columns):
                chart_type = "line"

        # Check if query suggests comparison
        if any(word in query.lower() for word in ["compare", "comparison", "versus", "vs"]):
            chart_type = "bar"

        # Check if query suggests distribution
        if any(word in query.lower() for word in ["distribution", "spread", "histogram"]):
            chart_type = "histogram"

        # Check if query suggests relationship
        if any(word in query.lower() for word in ["relationship", "correlation", "scatter"]):
            chart_type = "scatter"

        # Check if query suggests part-to-whole
        if any(word in query.lower() for word in ["percentage", "proportion", "share", "pie"]):
            chart_type = "pie"

        # Generate appropriate chart
        if chart_type == "bar":
            if len(df.columns) >= 2:
                # Use first string/categorical column as x
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

                if categorical_cols and numerical_cols:
                    x_col = categorical_cols[0]
                    y_col = numerical_cols[0]
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    return fig
                elif len(numerical_cols) >= 2:
                    x_col = numerical_cols[0]
                    y_col = numerical_cols[1]
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    return fig
                else:
                    # Fallback: Use index and first column
                    fig = px.bar(df, title="Query Results")
                    return fig
            else:
                # Single column - show counts
                fig = px.bar(df.iloc[:, 0].value_counts(), title="Count by Value")
                return fig

        elif chart_type == "line":
            date_cols = [col for col in df.columns if any(term in col.lower() for term in ["date", "time", "year", "month", "day"])]
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            if date_cols and numerical_cols:
                fig = px.line(df, x=date_cols[0], y=numerical_cols[0], title=f"{numerical_cols[0]} over Time")
                return fig
            elif len(numerical_cols) >= 2:
                fig = px.line(df, x=numerical_cols[0], y=numerical_cols[1], title=f"Trend Analysis")
                return fig
            else:
                # Fallback
                fig = px.line(df, title="Trend Analysis")
                return fig

        elif chart_type == "pie":
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            if categorical_cols and numerical_cols:
                fig = px.pie(df, names=categorical_cols[0], values=numerical_cols[0],
                            title=f"{numerical_cols[0]} Distribution by {categorical_cols[0]}")
                return fig
            else:
                # Fallback
                fig = px.pie(df, title="Distribution")
                return fig

        elif chart_type == "scatter":
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            if len(numerical_cols) >= 2:
                fig = px.scatter(df, x=numerical_cols[0], y=numerical_cols[1],
                                title=f"{numerical_cols[1]} vs {numerical_cols[0]}")
                return fig
            else:
                # Fallback
                fig = px.scatter(df, title="Relationship Analysis")
                return fig

        elif chart_type == "histogram":
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            if numerical_cols:
                fig = px.histogram(df, x=numerical_cols[0], title=f"Distribution of {numerical_cols[0]}")
                return fig
            else:
                # Fallback
                fig = px.histogram(df, title="Distribution Analysis")
                return fig

        # Fallback to simple bar chart if all else fails
        return px.bar(df, title="Query Results")

    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return None

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if "sql" in message:
            with st.expander("View SQL"):
                st.code(message["sql"], language="sql")
        st.markdown(message["content"])

        if "data" in message and isinstance(message["data"], pd.DataFrame):
            with st.expander("View Data"):
                st.dataframe(message["data"].head(10))

        if message.get("fig"):
            st.plotly_chart(message.get("fig"), use_container_width=True)

        # if message.get("viz_data") and "data" in message and isinstance(message["data"], pd.DataFrame):
        #     fig = generate_visualization(message["data"], message.get("query", ""))
        #     if fig is not None:
        #         st.plotly_chart(fig, use_container_width=True)
        #     else:
        #         st.info("Unable to generate visualization for this data.")

        # Display data visualization button if this message has data
        if message.get("has_data"): #and idx == len(st.session_state.messages) - 1:
            message_id = f"msg_{idx}"






            # Check if this message already has visualization
            # if message_id not in st.session_state.visualize_data:
            #     if st.button("📊 Visualize Data", key=f"viz_btn_{idx}"):
            #         st.session_state.visualize_data[message_id] = True
            #
            # # Show visualization if requested
            if message_id in st.session_state.visualize_data:
                data = message.get("data")
                if data is not None and isinstance(data, pd.DataFrame):
                    fig = generate_visualization(data, message.get("query", ""))
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Unable to generate visualization for this data.")

# Get user input
user_query = st.chat_input("Ask about documents or data...")

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
                # Route the query to the appropriate agent
                agent_type = st.session_state.agent_router.route_query(user_query)

                if agent_type == "duckdb":
                    # Use DuckDB agent for structured data queries
                    with st.info("Processing structured data query..."):
                        result = st.session_state.duckdb_agent.process_query(user_query)

                    if result["type"] == "duckdb_result":
                        # Show SQL query
                        with st.expander("View SQL"):
                            st.code(result["sql"], language="sql")

                        # Show data summary
                        st.markdown(result["summary"])

                        # Show limited preview of data
                        with st.expander("View Data"):
                            st.dataframe(result["data"].head(10))

                        viz_data = st.button("📊 Visualize Data")
                        # Visualization button
                        # if st.button("📊 Visualize Data"):
                        #     fig = generate_visualization(result["data"], user_query)
                        #     logger.info(f"the figure is {fig}")
                        #     if fig is not None:
                        #         st.plotly_chart(fig, use_container_width=True)
                        #     else:
                        #         st.info("Unable to generate visualization for this data.")
                        # Prepare response for chat history
                        response_content = f"""**Data Query Result:**

{result["summary"]}

_Query returned {result['row_count']} rows and {result['column_count']} columns._"""

                        # Add response to chat history with data for visualization
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_content,
                            "has_data": True,
                            "data": result["data"],
                            "query": user_query,
                            "viz_data": viz_data,
                            "sql": result["sql"]
                        })
                    else:
                        # Handle error
                        error_message = f"Error processing data query: {result.get('error', 'Unknown error')}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

                else:  # RAG agent
                    # Use RAG chain for document queries
                    st.info("Processing document query...")
                    ai_response = st.session_state.chain(user_query)
                    st.markdown(ai_response)

                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})

            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add footer
st.markdown("---")
st.markdown("Multi-Agent Intelligent Assistant powered by LangChain, DuckDB, and Groq")
