import os
import psycopg2
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

# Load environment variables
load_dotenv()

# API Keys
groq_api_key = os.getenv("GROQ_API_KEY")
hf_api = os.getenv("HUGGINGFACE_API_KEY")

# Model setup
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=groq_api_key)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# PostgreSQL connection
connection = "postgresql+psycopg2://postgres:cicada3301@127.0.0.1:5432/sales_data"
collection_name = "sales_data_embeddings"

# Vector store
vector_store = PGVector(
    embeddings=embedding_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

### **STEP 2: Try similarity search**
query_text = "Amul"
query_embedding = embedding_model.embed_query(query_text)

results = vector_store.similarity_search_by_vector(query_embedding, k=10)

if not results:
    print("⚠️ No relevant matches found for the query. Try a different keyword.")
else:
    print("✅ Found matches:")
    for doc in results:
        print(doc.page_content)

### **STEP 3: Use MMR (Not BM25)**
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
docs = retriever.invoke(query_text)

print("\n🔍 MMR Results:")
for doc in docs:
    print(doc.page_content)
