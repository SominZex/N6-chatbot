import pymysql
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from src.common import get_db_connection
import re

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("./data/faiss_index", embeddings, allow_dangerous_deserialization=True)

llm = LlamaCpp(
    model_path="./data/model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    temperature=0.1,
    max_tokens=256,
    n_ctx=4096,
    n_gpu_layers=30,
    verbose=False
)

sql_llm = LlamaCpp(
    model_path="./data/model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    temperature=0.1,
    max_tokens=128,
    n_ctx=4096,
    n_gpu_layers=30,
    top_k=40,
    top_p=0.95,
    verbose=False   
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

print("🔥 RAG System Ready!")

db = SQLDatabase.from_uri("mysql+pymysql://root@localhost/sales_data")
sql_chain = create_sql_query_chain(sql_llm, db)

def is_sql_query(question):
    """
    Smarter detection of SQL-related queries using both keywords and patterns
    """
    quick_keywords = {"select", "count", "sum", "average", "total", "how many"}
    if not any(keyword in question.lower() for keyword in quick_keywords):
        return False
    
    strong_sql_indicators = [
        r"select.*from",
        r"count.*where",
        r"sum.*where",
        r"average.*where",
        r"total number of",
        r"how many.*are there",
        r"list all",
        r"show me all",
        r"find.*where",
    ]
    
    question_lower = question.lower()
    for pattern in strong_sql_indicators:
        if re.search(pattern, question_lower):
            return True
            
    sql_context_keywords = {
        "database": 5,
        "table": 5,
        "records": 5,
        "entries": 5,
        "store": 3,
        "sales": 3,
        "total": 3,
        "count": 3,
        "average": 3,
        "sum": 3,
    }
    
    score = sum(weight for keyword, weight in sql_context_keywords.items() 
                if keyword in question_lower)
            
    return score >= 5

def format_sql_results(results):
    """
    Format SQL results for better readability
    """
    if not results:
        return "No results found."
    
    if len(results) == 1 and len(results[0]) == 1:
        return f"Result: {results[0][0]}"
    
    return results

def execute_sql_query(query):
    """
    Execute SQL query with improved error handling
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return format_sql_results(results)
        
    except pymysql.Error as e:
        error_msg = str(e)
        if "Unknown column" in error_msg or "Table" in error_msg:
            raise Exception(f"SQL Error: {error_msg}")
        print(f"SQL Warning: {error_msg}")
        return format_sql_results(()) 
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def ask_question(query):
    if is_sql_query(query):
        print("🔍 Query detected as SQL-related. Generating & executing SQL...")
        try:
            sql_query = sql_chain.invoke({"question": query})
            
            if not isinstance(sql_query, str):
                print("⚠️ Unexpected SQL response format, falling back to RAG...")
                response = rag_chain.invoke(query)
                return response["result"]
            
            print(f"📝 Generated SQL Query: {sql_query}")
            
            result = execute_sql_query(sql_query)
            if isinstance(result, str) and result.startswith("SQL Error"):
                raise Exception(result)
            return result
            
        except Exception as e:
            if "SQL Error" in str(e):
                print(f"⚠️ {str(e)}, falling back to RAG...")
            else:
                print(f"⚠️ Error processing SQL query: {str(e)}, falling back to RAG...")
            response = rag_chain.invoke(query)
            return response["result"]
    
    try:
        response = rag_chain.invoke(query)
        return response["result"]
    except Exception as e:
        return f"Error processing RAG query: {str(e)}"