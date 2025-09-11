"""
DuckDB agent module for handling structured data queries.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import duckdb
import mlflow
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import config

from logger import Logger

# Configure logging
logger = Logger()

class DuckDBAgent:
    """
    Agent for handling structured data queries using DuckDB.
    Generates SQL from natural language and executes it.
    """

    def __init__(self, db_path: Optional[str] = None, llm=None):
        """
        Initialize the DuckDB agent.

        Args:
            db_path: Optional path to DuckDB database file
            llm: Optional LLM to use for SQL generation
        """
        self.db_path = db_path
        self.llm = llm or self._initialize_llm()
        self.conn = self._initialize_connection()

        # Initialize MLFlow tracking
        try:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("duckdb_agent")
        except Exception as e:
            logger.error(f"Error initializing MLFlow: {str(e)}")

    def _initialize_llm(self):
        """Initialize the LLM for SQL generation"""
        try:
            return ChatGroq(
                groq_api_key=config.GROQ_API_KEY,
                model_name=config.GROQ_CHAT_MODEL
            )
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {str(e)}")
            raise

    def _initialize_connection(self):
        """Initialize connection to DuckDB"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # Connect to DuckDB
            conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to DuckDB at {self.db_path}")
            return conn
        except Exception as e:
            logger.error(f"Error connecting to DuckDB: {str(e)}")
            raise

    def get_available_tables(self) -> List[str]:
        """
        Get list of available tables in the database.

        Returns:
            List of table names
        """
        try:
            result = self.conn.execute("SHOW TABLES").fetchall()
            tables = [row[0] for row in result]
            return tables
        except Exception as e:
            logger.error(f"Error getting tables: {str(e)}")
            return []

    def get_table_schema(self, table_name: str) -> str:
        """
        Get schema information for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            String representation of table schema
        """
        try:
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            schema = "\n".join([f"{row[0]} {row[1]}" for row in result])
            return schema
        except Exception as e:
            logger.error(f"Error getting schema for {table_name}: {str(e)}")
            return ""

    def get_all_schemas(self) -> str:
        """
        Get schema information for all tables.

        Returns:
            String representation of all table schemas
        """
        schemas = []
        for table in self.get_available_tables():
            schema = self.get_table_schema(table)
            schemas.append(f"Table: {table}\n{schema}")

        return "\n\n".join(schemas)

    def generate_sql(self, query: str) -> str:
        """
        Generate SQL from natural language query.

        Args:
            query: Natural language query

        Returns:
            Generated SQL query
        """
        with mlflow.start_run(run_name="sql_generation"):
            mlflow.log_param("user_query", query)

            # Get database schema information
            schemas = self.get_all_schemas()

            prompt = f"""You are a SQL expert. Given the following database schema and a question, 
generate a DuckDB SQL query that answers the question.

DATABASE SCHEMA:
{schemas}

USER QUESTION: {query}

Respond ONLY with the SQL query and nothing else. Do not include any explanations, just the SQL.
The SQL should be syntactically correct for DuckDB."""

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                sql = response.content.strip()

                # Extract SQL if enclosed in backticks
                if "```sql" in sql:
                    sql = sql.split("```sql")[1].split("```")[0].strip()
                elif "```" in sql:
                    sql = sql.split("```")[1].split("```")[0].strip()

                mlflow.log_param("generated_sql", sql)
                logger.info(f"Generated SQL: {sql}")
                return sql

            except Exception as e:
                logger.error(f"Error generating SQL: {str(e)}")
                mlflow.log_param("error", str(e))
                raise

    def execute_sql(self, sql: str) -> Tuple[pd.DataFrame, str]:
        """
        Execute SQL query and return results.

        Args:
            sql: SQL query to execute

        Returns:
            Tuple of (DataFrame with results, summary text)
        """
        with mlflow.start_run(run_name="sql_execution"):
            mlflow.log_param("sql_query", sql)

            try:
                # Execute the query
                result = self.conn.execute(sql).fetchdf()

                # Generate a summary of the results
                summary = self._generate_summary(result, sql)

                # Log metrics
                mlflow.log_metric("result_row_count", len(result))
                mlflow.log_metric("result_column_count", len(result.columns))

                return result, summary

            except Exception as e:
                logger.error(f"Error executing SQL: {str(e)}")
                mlflow.log_param("error", str(e))
                raise

    def _generate_summary(self, df: pd.DataFrame, sql: str) -> str:
        """
        Generate a summary of the query results.

        Args:
            df: DataFrame with query results
            sql: SQL query that was executed

        Returns:
            Summary text
        """
        try:
            # Basic statistics about the result
            row_count = len(df)
            col_count = len(df.columns)

            prompt = f"""You are a data analyst. Summarize the following SQL query results in a clear, concise manner.

SQL Query: {sql}

Result Statistics:
- {row_count} rows returned
- {col_count} columns: {', '.join(df.columns.tolist())}

First few rows of data:
{df.head(5).to_string()}

Provide a concise summary of what this data shows. Include key insights, trends, or notable findings if applicable.
Keep your response to 3-5 sentences."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip()

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Query returned {len(df)} rows with {len(df.columns)} columns."

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query end-to-end.

        Args:
            query: Natural language query

        Returns:
            Dictionary with results and metadata
        """
        try:
            # Generate SQL
            sql = self.generate_sql(query)

            # Execute SQL
            result_df, summary = self.execute_sql(sql)

            return {
                "type": "duckdb_result",
                "query": query,
                "sql": sql,
                "data": result_df,
                "summary": summary,
                "row_count": len(result_df),
                "column_count": len(result_df.columns)
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "type": "error",
                "query": query,
                "error": str(e)
            }

    def create_sample_data(self):
        """Create sample tables with data for testing"""
        try:
            # Create sales table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER,
                date DATE,
                product_id INTEGER,
                customer_id INTEGER,
                quantity INTEGER,
                price DECIMAL(10,2),
                total DECIMAL(10,2)
            )
            """)

            # Insert sample data
            self.conn.execute("""
            INSERT OR REPLACE INTO sales VALUES
            (1, '2023-01-01', 101, 1, 2, 10.99, 21.98),
            (2, '2023-01-02', 102, 2, 1, 24.99, 24.99),
            (3, '2023-01-03', 103, 3, 3, 5.99, 17.97),
            (4, '2023-01-04', 101, 4, 1, 10.99, 10.99),
            (5, '2023-01-05', 104, 5, 2, 15.99, 31.98),
            (6, '2023-01-06', 102, 1, 1, 24.99, 24.99),
            (7, '2023-01-07', 103, 2, 2, 5.99, 11.98),
            (8, '2023-01-08', 104, 3, 1, 15.99, 15.99),
            (9, '2023-01-09', 101, 4, 3, 10.99, 32.97),
            (10, '2023-01-10', 102, 5, 1, 24.99, 24.99)
            """)

            # Create products table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER,
                name VARCHAR,
                category VARCHAR,
                price DECIMAL(10,2)
            )
            """)

            # Insert product data
            self.conn.execute("""
            INSERT OR REPLACE INTO products VALUES
            (101, 'Basic T-Shirt', 'Clothing', 10.99),
            (102, 'Premium Jacket', 'Clothing', 24.99),
            (103, 'Pen Set', 'Office Supplies', 5.99),
            (104, 'Notebook', 'Office Supplies', 15.99),
            (105, 'Coffee Mug', 'Kitchen', 7.99)
            """)

            # Create customers table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER,
                name VARCHAR,
                email VARCHAR,
                country VARCHAR
            )
            """)

            # Insert customer data
            self.conn.execute("""
            INSERT OR REPLACE INTO customers VALUES
            (1, 'John Doe', 'john@example.com', 'USA'),
            (2, 'Jane Smith', 'jane@example.com', 'Canada'),
            (3, 'Bob Johnson', 'bob@example.com', 'UK'),
            (4, 'Alice Brown', 'alice@example.com', 'Australia'),
            (5, 'Charlie Wilson', 'charlie@example.com', 'Germany')
            """)

            logger.info("Sample data created successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
            return False

