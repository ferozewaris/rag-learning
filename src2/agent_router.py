"""
Agent router module for determining which agent to use for a given query.
"""
import logging
from typing import Literal, Dict, Any, Optional
import mlflow
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRouter:
    """
    Router for determining which agent should handle a user query.
    Can route to RAG agent for document-based queries or DuckDB agent for structured data queries.
    """

    def __init__(self, llm=None):
        """
        Initialize the agent router.

        Args:
            llm: Optional LLM to use for routing decisions. If None, defaults to Groq
        """
        self.llm = llm or self._initialize_llm()

        # Initialize MLFlow tracking
        try:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("agent_router")
        except Exception as e:
            logger.error(f"Error initializing MLFlow: {str(e)}")

    def _initialize_llm(self):
        """Initialize the LLM for routing decisions"""
        try:
            return ChatGroq(
                groq_api_key=config.GROQ_API_KEY,
                model_name=config.GROQ_CHAT_MODEL
            )
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {str(e)}")
            raise

    def route_query(self, query: str) -> Literal["rag", "duckdb"]:
        """
        Determine which agent should handle the query.

        Args:
            query: User query string

        Returns:
            "rag" for document queries or "duckdb" for structured data queries
        """
        # Log the routing request
        with mlflow.start_run(run_name="agent_routing"):
            mlflow.log_param("query", query)

            # Use LLM to classify the query
            prompt = f"""Determine if the following query requires:
1. Searching through documents and retrieving information (RAG)
2. Analyzing structured data with SQL-like operations (DuckDB)

Query: {query}

Answer with only one word: "rag" or "duckdb"."""

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                result = response.content.strip().lower()

                # Handle potential invalid responses
                if result not in ["rag", "duckdb"]:
                    if "sql" in result.lower() or "database" in result.lower() or "data" in result.lower():
                        result = "duckdb"
                    else:
                        result = "rag"

                mlflow.log_param("routing_decision", result)
                logger.info(f"Routing decision for query '{query}': {result}")
                return result

            except Exception as e:
                logger.error(f"Error determining agent: {str(e)}")
                # Default to RAG if there's an error
                mlflow.log_param("routing_decision", "rag")
                mlflow.log_param("error", str(e))
                return "rag"

    def route_with_rules(self, query: str) -> Literal["rag", "duckdb"]:
        """
        Fallback rule-based router in case the LLM router fails.

        Args:
            query: User query string

        Returns:
            "rag" for document queries or "duckdb" for structured data queries
        """
        # Simple rule-based approach as fallback
        data_keywords = [
            "database", "sql", "query", "table", "data", "analyze",
            "statistics", "trend", "count", "average", "sum", "group by",
            "filter", "compare", "visualization", "chart", "graph", "plot",
            "dashboard", "metrics", "kpi", "sales", "numbers", "percentage"
        ]

        # Check if the query contains data-related keywords
        if any(keyword in query.lower() for keyword in data_keywords):
            return "duckdb"

        return "rag"
