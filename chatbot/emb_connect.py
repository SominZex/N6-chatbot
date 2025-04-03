import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=groq_api_key)

DB_CONFIG = {
    "dbname": "sales_data",
    "user": "postgres",
    "password": "cicada3301",
    "host": "localhost",
    "port": 5432
}

# Connect to PostgreSQL
def connect_db():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    print("✅ Connected to PostgreSQL")
    return conn

conn = connect_db()

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate normalized embeddings"""
    return embedding_model.encode(text, normalize_embeddings=True).tolist()

def find_relevant_vectors(user_query, top_k=3):
    """Find the most relevant text embeddings"""
    query_embedding = get_embedding(user_query)

    with conn.cursor() as cur:
        query = """
            SELECT combined_text, embedding <=> %s::vector AS distance
            FROM sales_data_embeddings  
            ORDER BY distance ASC
            LIMIT %s;
        """
        cur.execute(query, (query_embedding, top_k))
        results = cur.fetchall()

    return [r[0] for r in results] if results else []

def generate_response(user_query):
    """Retrieve relevant context and generate response"""
    relevant_docs = find_relevant_vectors(user_query)
    print("🔍 Relevant Documents:", relevant_docs)

    context = "\n".join(relevant_docs) if relevant_docs else "No relevant context found."

    prompt = f"""User: {user_query}
    Relevant Context: {context}
    Assistant:"""

    response = llm.invoke(prompt)
    return response.content

# Chat loop
try:
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break
        response = generate_response(user_input)
        print(f"Bot: {response}")
except KeyboardInterrupt:
    print("\n👋 Exiting gracefully...")
finally:
    conn.close()
    print("🔒 Database connection closed.")
