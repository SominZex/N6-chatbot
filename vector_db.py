import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables (create a .env file with these variables)
load_dotenv()

# Database connection parameters
DB_NAME = os.getenv("DB_NAME", "sales_data")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "cicada3301")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
EMBEDDING_DIMENSION = 384  

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

def setup_pgvector():
    """Set up pgvector extension in PostgreSQL"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create pgvector extension if it doesn't exist
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    conn.close()
    print("pgvector extension installed!")

def create_embedding_table():
    """Create table to store embeddings with additional indices"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create table for embeddings
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS sales_data_embeddings (
        id SERIAL PRIMARY KEY,
        original_row_id INTEGER,
        combined_text TEXT,
        embedding vector({EMBEDDING_DIMENSION}),
        
        -- Order info
        orderdate DATE,
        time TIME WITHOUT TIME ZONE,
        orderstatus VARCHAR(255),
        ordertype VARCHAR(255),
        orderfrom VARCHAR(255),
        invoice VARCHAR(255),
        storeinvoice VARCHAR(255),
        
        -- Product info
        productid BIGINT,
        productname VARCHAR(255),
        barcode VARCHAR(255),
        hsncode VARCHAR(255),
        description TEXT,
        brandname VARCHAR(255),
        categoryname VARCHAR(255),
        subcategoryof VARCHAR(255),
        
        -- Financial info
        quantity BIGINT,
        sellingprice NUMERIC,
        costprice NUMERIC,
        discountamount NUMERIC,
        totalproductprice NUMERIC,
        orderamountnet NUMERIC,
        orderamounttax NUMERIC,
        paymentmethod VARCHAR(255),
        
        -- Store and customer info
        storename VARCHAR(255),
        gstin VARCHAR(255),
        customername VARCHAR(255),
        customernumber VARCHAR(255)
    )
    """)
    conn.commit()
    
    # Create vector search index for faster similarity searches using HNSW
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_embedding_hnsw ON sales_data_embeddings USING hnsw (embedding vector_cosine_ops)")
        print("Vector search index created!")
    except Exception as e:
        print(f"Note on vector index: {e}")
        conn.rollback()
    
    # Add additional indices for commonly filtered columns
    indices = [
        # Date and time filters
        "CREATE INDEX IF NOT EXISTS idx_sales_orderdate ON sales_data_embeddings (orderdate)",
        
        # Product filters
        "CREATE INDEX IF NOT EXISTS idx_sales_productid ON sales_data_embeddings (productid)",
        "CREATE INDEX IF NOT EXISTS idx_sales_productname ON sales_data_embeddings USING gin (productname gin_trgm_ops)",
        "CREATE INDEX IF NOT EXISTS idx_sales_brandname ON sales_data_embeddings USING gin (brandname gin_trgm_ops)",
        "CREATE INDEX IF NOT EXISTS idx_sales_categoryname ON sales_data_embeddings USING gin (categoryname gin_trgm_ops)",
        
        # Order status and type
        "CREATE INDEX IF NOT EXISTS idx_sales_orderstatus ON sales_data_embeddings (orderstatus)",
        "CREATE INDEX IF NOT EXISTS idx_sales_ordertype ON sales_data_embeddings (ordertype)",
        "CREATE INDEX IF NOT EXISTS idx_sales_paymentmethod ON sales_data_embeddings (paymentmethod)",
        
        # Price range filters
        "CREATE INDEX IF NOT EXISTS idx_sales_sellingprice ON sales_data_embeddings (sellingprice)",
        "CREATE INDEX IF NOT EXISTS idx_sales_orderamountnet ON sales_data_embeddings (orderamountnet)",
        
        # Store and customer
        "CREATE INDEX IF NOT EXISTS idx_sales_storename ON sales_data_embeddings USING gin (storename gin_trgm_ops)",
        "CREATE INDEX IF NOT EXISTS idx_sales_customername ON sales_data_embeddings USING gin (customername gin_trgm_ops)",
        "CREATE INDEX IF NOT EXISTS idx_sales_customernumber ON sales_data_embeddings (customernumber)"
    ]
    
    # Make sure trigram extension is installed for text search
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        print("Trigram extension installed for text search!")
    except Exception as e:
        print(f"Note on trigram extension: {e}")
        conn.rollback()
    
    # Create all the indices
    for idx, index_sql in enumerate(indices):
        try:
            print(f"Creating index {idx+1}/{len(indices)}...")
            cur.execute(index_sql)
            conn.commit()
        except Exception as e:
            print(f"Note on index {idx+1}: {e}")
            conn.rollback()
    
    conn.close()
    print("Embedding table and all indices created!")

def prepare_text_for_embedding(row):
    """Combines relevant text fields into a single string for embedding"""
    text_fields = []
    
    # Add product information
    if pd.notna(row.get('productname')):
        text_fields.append(f"Product: {row['productname']}")
    if pd.notna(row.get('description')):
        text_fields.append(f"Description: {row['description']}")
    if pd.notna(row.get('brandname')):
        text_fields.append(f"Brand: {row['brandname']}")
    if pd.notna(row.get('categoryname')):
        text_fields.append(f"Category: {row['categoryname']}")
    if pd.notna(row.get('subcategoryof')):
        text_fields.append(f"Subcategory: {row['subcategoryof']}")
        
    # Add order information
    if pd.notna(row.get('orderstatus')):
        text_fields.append(f"Status: {row['orderstatus']}")
    if pd.notna(row.get('ordertype')):
        text_fields.append(f"Order Type: {row['ordertype']}")
    if pd.notna(row.get('orderfrom')):
        text_fields.append(f"Order From: {row['orderfrom']}")
    if pd.notna(row.get('paymentmethod')):
        text_fields.append(f"Payment Method: {row['paymentmethod']}")
        
    # Add store and customer information
    if pd.notna(row.get('storename')):
        text_fields.append(f"Store: {row['storename']}")
    if pd.notna(row.get('customername')):
        text_fields.append(f"Customer: {row['customername']}")
    
    return " ".join(filter(None, text_fields))

def generate_and_store_embeddings():
    """Generate embeddings for sales data and store them in PostgreSQL"""
    # Load data from the sales_data table
    conn = get_db_connection()
    print("Loading data from sales_data table...")
    query = "SELECT * FROM sales_data"
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} rows from sales_data table")
    
    # Initialize the embedding model
    print(f"Initializing embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Generate combined text for embedding
    print("Preparing text for embeddings...")
    df['combined_text'] = df.apply(prepare_text_for_embedding, axis=1)
    
    # Process in batches to avoid memory issues with large datasets
    batch_size = 500
    total_rows = len(df)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Clear existing data (optional - comment out if you want to keep existing data)
    print("Clearing existing embeddings...")
    cur.execute("TRUNCATE TABLE sales_data_embeddings")
    conn.commit()
    
    print(f"Generating embeddings in batches of {batch_size}...")
    for i in range(0, total_rows, batch_size):
        end = min(i + batch_size, total_rows)
        batch = df.iloc[i:end]
        
        print(f"Processing batch {i+1} to {end} of {total_rows} rows")
        
        # Generate embeddings for this batch
        batch_text = batch['combined_text'].tolist()
        batch_embeddings = model.encode(batch_text)
        
        # Prepare data for bulk insert
        batch_data = []
        for idx, row in batch.iterrows():
            embedding = batch_embeddings[batch.index.get_loc(idx)].tolist()
            
            # Collect all fields to store
            data_row = [
                idx,  # original_row_id
                row.get('combined_text'),
                embedding
            ]
            
            # Add all other columns
            for col in [
                'orderdate', 'time', 'orderstatus', 'ordertype', 'orderfrom', 
                'invoice', 'storeinvoice', 'productid', 'productname', 'barcode', 
                'hsncode', 'description', 'brandname', 'categoryname', 'subcategoryof', 
                'quantity', 'sellingprice', 'costprice', 'discountamount', 
                'totalproductprice', 'orderamountnet', 'orderamounttax', 'paymentmethod', 
                'storename', 'gstin', 'customername', 'customernumber'
            ]:
                data_row.append(row.get(col))
                
            batch_data.append(tuple(data_row))
        
        # Create the placeholders for the SQL query
        placeholders = ', '.join(['%s'] * len(batch_data[0]))
        columns = """
            original_row_id, combined_text, embedding, 
            orderdate, time, orderstatus, ordertype, orderfrom, 
            invoice, storeinvoice, productid, productname, barcode, 
            hsncode, description, brandname, categoryname, subcategoryof, 
            quantity, sellingprice, costprice, discountamount, 
            totalproductprice, orderamountnet, orderamounttax, paymentmethod, 
            storename, gstin, customername, customernumber
        """
        
        # Execute bulk insert
        insert_query = f"""
        INSERT INTO sales_data_embeddings (
            {columns}
        ) VALUES ({placeholders})
        """
        
        execute_values(cur, insert_query, batch_data)
        conn.commit()
        print(f"Batch {i+1}-{end} inserted successfully!")
    
    conn.close()
    print("All embeddings generated and stored successfully!")

def test_vector_search(query_text="high quality branded product", filter_params=None, top_n=5):
    """Test vector search functionality with filters"""
    print(f"Testing vector search with query: '{query_text}'")
    
    if filter_params is None:
        filter_params = {}
        
    print(f"Using filters: {filter_params}")
    
    # Create embedding for the query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query_text).tolist()
    
    # Connect to database
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Register vector type with psycopg2
    from pgvector.psycopg2 import register_vector
    register_vector(conn)
    
    # Prepare SQL query with optional filters
    sql = """
    SELECT 
        id, 
        productname,
        brandname,
        categoryname,
        sellingprice,
        orderdate,
        combined_text,
        1 - (embedding <=> %s) as similarity
    FROM sales_data_embeddings
    """
    
    params = [query_embedding]
    
    # Add filters if provided
    where_clauses = []
    if filter_params:
        for key, value in filter_params.items():
            if key == 'orderdate':
                if isinstance(value, dict) and 'start' in value and 'end' in value:
                    where_clauses.append(f"{key} BETWEEN %s AND %s")
                    params.extend([value['start'], value['end']])
                else:
                    where_clauses.append(f"{key} = %s")
                    params.append(value)
            elif key in ['sellingprice', 'orderamountnet', 'quantity']:
                if isinstance(value, dict) and 'min' in value and 'max' in value:
                    where_clauses.append(f"{key} BETWEEN %s AND %s")
                    params.extend([value['min'], value['max']])
                else:
                    where_clauses.append(f"{key} = %s")
                    params.append(value)
            else:
                if isinstance(value, str):
                    # For text fields, use trigram similarity
                    where_clauses.append(f"{key} ILIKE %s")
                    params.append(f"%{value}%")
                else:
                    where_clauses.append(f"{key} = %s")
                    params.append(value)
    
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    
    # Add ordering and limit
    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params.extend([query_embedding, top_n])
    
    # Execute the query
    cur.execute(sql, params)
    results = cur.fetchall()
    
    print("\nSearch Results:")
    for row in results:
        print(f"ID: {row[0]}")
        print(f"Product: {row[1]}")
        print(f"Brand: {row[2]}")
        print(f"Category: {row[3]}")
        print(f"Price: {row[4]}")
        print(f"Date: {row[5]}")
        print(f"Text: {row[6][:100]}...")  
        print(f"Similarity: {row[7]:.4f}")
        print("-" * 50)
    
    # Analyze query performance
    analyze_sql = "EXPLAIN ANALYZE " + sql
    print("\nQuery Performance Analysis:")
    cur.execute(analyze_sql, params)
    analysis = cur.fetchall()
    for line in analysis:
        print(line[0])
    
    conn.close()

def main():
    print("Starting vector embedding process with improved indexing...")
    
    # Set up pgvector extension
    setup_pgvector()
    
    # Create table for embeddings with additional indices
    create_embedding_table()
    
    # Generate and store embeddings
    generate_and_store_embeddings()
    
    # Test basic vector search
    test_vector_search("high quality branded product")
    
    # Test vector search with filters
    filter_example = {
        'brandname': 'adidas', 
        'sellingprice': {'min': 1000, 'max': 5000}, 
        'orderdate': {'start': '2022-01-01', 'end': '2023-12-31'} 
    }
    test_vector_search("athletic shoes", filter_example)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()