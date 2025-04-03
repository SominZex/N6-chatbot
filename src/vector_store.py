import os
import json
from decimal import Decimal  
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from db_connector import fetch_sales_data, get_column_names 


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

sales_data = fetch_sales_data()

column_names = get_column_names() 

if not column_names:
    raise ValueError("No column names found. Check your database schema function.")

if not sales_data:
    raise ValueError("No sales data found. Check your database query.")

sales_data_dicts = [dict(zip(column_names, row)) for row in sales_data]

def serialize_row(row):
    """Convert non-serializable data types before JSON serialization."""
    return {
        key: (
            value.isoformat() if hasattr(value, "isoformat") else 
            float(value) if isinstance(value, Decimal) else  
            value
        )
        for key, value in row.items()
    }

# Process Sales Data for FAISS
documents = [Document(page_content=json.dumps(serialize_row(row))) for row in sales_data_dicts]

# Load Database Schema Information
schema_path = "./data/schema_info.txt"
if os.path.exists(schema_path):
    with open(schema_path, "r") as f:
        schema_info = f.read()
    schema_doc = Document(page_content=f"Database Schema: {schema_info}")
else:
    schema_doc = Document(page_content="Database Schema: No schema info available.")

vectorstore = FAISS.from_documents(documents + [schema_doc], embeddings)

# Save FAISS Index
vectorstore.save_local("./data/faiss_index")
