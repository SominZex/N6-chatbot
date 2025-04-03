import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_NAME = os.getenv("DB_NAME", "sales_data")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "cicada3301")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 128  
CHUNK_SIZE = 100000  
INSERT_BATCH_SIZE = 5000  

def get_db_connection():
    """Establish a PostgreSQL database connection."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def prepare_text_for_embedding(row):
    """Prepare text fields for embedding by concatenating relevant columns."""
    text_fields = []
    
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
    if pd.notna(row.get('orderstatus')):
        text_fields.append(f"Status: {row['orderstatus']}")
    if pd.notna(row.get('ordertype')):
        text_fields.append(f"Order Type: {row['ordertype']}")
    if pd.notna(row.get('orderfrom')):
        text_fields.append(f"Order From: {row['orderfrom']}")
    if pd.notna(row.get('paymentmethod')):
        text_fields.append(f"Payment Method: {row['paymentmethod']}")
    if pd.notna(row.get('storename')):
        text_fields.append(f"Store: {row['storename']}")
    if pd.notna(row.get('customername')):
        text_fields.append(f"Customer: {row['customername']}")

    return " ".join(text_fields)

def process_and_store_embeddings(df, model):
    """Generate embeddings and insert them into PostgreSQL."""
    print(f"Processing {len(df)} rows...")

    df.rename(columns=lambda x: x.strip(), inplace=True)
    df["combined_text"] = df.apply(prepare_text_for_embedding, axis=1)
    
    print("Generating embeddings...")
    embeddings = model.encode(df["combined_text"].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
    df["embedding"] = [emb.tolist() for emb in embeddings]

    values = [
        (
            row["combined_text"],
            np.array(row["embedding"]).tolist(),
            row["orderdate"],
            row["time"],
            row["orderstatus"],
            row["ordertype"],
            row["orderfrom"],
            row["invoice"],
            row["storeinvoice"],
            row["productid"],
            row["productname"],
            row["barcode"],
            row["hsncode"],
            row["description"],
            row["brandname"],
            row["categoryname"],
            row["subcategoryof"],
            row["quantity"],
            row["sellingprice"],
            row["costprice"],
            row["discountamount"],
            row["totalproductprice"],
            row["orderamountnet"],
            row["orderamounttax"],
            row["paymentmethod"],
            row["storename"],
            row["gstin"],
            row["customername"],
            row["customernumber"]
        )
        for _, row in df.iterrows()
    ]

    conn = get_db_connection()
    cur = conn.cursor()
    
    insert_query = """
    INSERT INTO sales_data_embeddings (combined_text, embedding, 
        orderdate, time, orderstatus, ordertype, orderfrom, 
        invoice, storeinvoice, productid, productname, barcode, hsncode, 
        description, brandname, categoryname, subcategoryof, quantity, 
        sellingprice, costprice, discountamount, totalproductprice, 
        orderamountnet, orderamounttax, paymentmethod, storename, 
        gstin, customername, customernumber
    ) VALUES %s
    """

    print("Inserting data into PostgreSQL...")
    try:
        for i in range(0, len(values), INSERT_BATCH_SIZE):
            batch = values[i:i + INSERT_BATCH_SIZE]
            execute_values(cur, insert_query, batch)
            conn.commit()
    except Exception as e:
        print("Error inserting batch:", e)
        conn.rollback()
    finally:
        cur.close()
        conn.close()
    
    print("Batch inserted successfully!")

def generate_and_store_embeddings():
    """Fetch data in chunks, generate embeddings, and insert them into PostgreSQL."""
    conn = get_db_connection()
    print(f"Fetching data in chunks of {CHUNK_SIZE} rows...")
    
    query = "SELECT * FROM sales_data"
    model = SentenceTransformer(EMBEDDING_MODEL) 

    try:
        for chunk in pd.read_sql(query, conn, chunksize=CHUNK_SIZE):
            process_and_store_embeddings(chunk, model)
    except Exception as e:
        print("Error during processing:", e)
    finally:
        conn.close()
    
    print("All data processed successfully!")

# Run the function
generate_and_store_embeddings()
