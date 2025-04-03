from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load API keys from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing Groq API Key. Set it as an environment variable.")

# PostgreSQL Connection Details
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "postgres",
    "password": "cicada3301",
    "dbname": "sales_data"
}
connection = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**DB_CONFIG)

# Initialize Embeddings Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to PGVector Vector Store
vector_store = PGVector(
    connection=connection,
    collection_name="sales_data_embeddings",
    embeddings=embedding_model
)

# Load LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# Define Similarity Search Function
def search_products(query_text, k=5):
    """Perform similarity search in PGVector."""
    docs = vector_store.similarity_search_with_score(query_text, k=k)
    
    if not docs:
        return "⚠️ No relevant matches found."
    
    results = [f"🔹 {doc.page_content} (Score: {score:.4f})" for doc, score in docs]
    return "\n".join(results)

# Define RAG-based Query Function
def rag_query(user_query, k=5):
    """Retrieve and generate response based on PGVector data."""
    retrieved_docs = search_products(user_query, k)
    
    if isinstance(retrieved_docs, str):
        return retrieved_docs

    prompt = PromptTemplate.from_template(
        """Context:
        {context}
        
        User Query: {query}
        
        Generate a relevant response based on the above context."""
    )
    
    response = llm.invoke(prompt.format(context=retrieved_docs, query=user_query))
    return response.content

# Command-line Chat Interface
def chat():
    print("🗨️ LLM Chatbot (Type 'exit' to quit)")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = rag_query(query, k=10)
        print("\n🤖 AI:", response)

if __name__ == "__main__":
    chat()
