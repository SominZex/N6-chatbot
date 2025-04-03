import os
import psycopg2
import numpy as np
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing Groq API Key. Set it as an environment variable.")

DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "postgres",
    "password": "cicada3301",
    "dbname": "sales_data"
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("✅ Connected to PostgreSQL")
except Exception as e:
    print(f"❌ Database Connection Failed: {e}")
    exit()

llm = ChatGroq(model_name="llama-3.3-70b-versatile")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_query_embedding(query_text):
    """
    Generates an embedding using HuggingFaceEmbeddings instead of Llama.
    """
    try:
        query_embedding = embedding_model.embed_query(query_text)
        return np.array(query_embedding, dtype=float)
    except Exception as e:
        raise ValueError(f"Embedding generation failed: {e}")

def similarity_search(query_text, k=20):
    """
    Performs a similarity search dynamically, ensuring all relevant products are retrieved.
    """
    query_embedding = get_query_embedding(query_text)
    
    search_query = """
    SELECT id, productname, combined_text, orderdate, 
        1 - (embedding <=> %s::vector) AS similarity
    FROM sales_data_embeddings  
    ORDER BY similarity DESC
    LIMIT %s;
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(search_query, (query_embedding.tolist(), k))
            results = cursor.fetchall()
        
        if not results:
            return "⚠️ No relevant matches found."

        search_results = [
            {
                "id": row[0],
                "product": row[1],
                "text": row[2],
                "date": row[3],
                "similarity": row[4]
            }
            for row in results
        ]

        print(f"🔍 Retrieved {len(search_results)} results")

        return search_results
    except Exception as e:
        return f"❌ Search error: {e}"

def rag_query(user_query, k=20):
    """
    Runs similarity search and enhances response using Llama 3.2.
    """
    print("🔍 Fetching relevant sales data...")
    reformatted_query = f"List all products related to {user_query.split()[-1]}" if "list" in user_query.lower() else user_query
    retrieved_docs = similarity_search(reformatted_query, k)

    if isinstance(retrieved_docs, str):
        return retrieved_docs

    context = "\n".join([f"{doc['product']}: {doc['text']}" for doc in retrieved_docs])

    print("🧠 Augmenting with Llama 3.2...")
    response = llm.invoke([
        HumanMessage(
            content=f"Context:\n{context}\n\nUser Query: {user_query}\n\nGenerate a relevant response."
        )
    ])
    
    return response.content

query = input("Query: ")
print(f"🔍 User Query: {query}")
response = rag_query(query, k=25) 
print("\n📢 AI Response:\n", response)
