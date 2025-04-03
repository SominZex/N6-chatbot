from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import mysql.connector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define model path
model_path = "./model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (Ensure correct device usage)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Connect to MySQL database
db_connection = mysql.connector.connect(    
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE")
)

# Custom function to retrieve data from MySQL
def retrieve_sales_data(product_name):
    cursor = db_connection.cursor()
    query = """
    SELECT SUM(totalProductPrice) AS total_sales
    FROM sales_data
    WHERE productName = %s
    AND orderDate >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH);
    """
    cursor.execute(query, (product_name,))
    result = cursor.fetchone()
    cursor.close()
    return result[0] if result else 0

# Example product name
product_name = "Red Bull Energy Drink, 350 Ml Can"

# Retrieve context using the product name
context = retrieve_sales_data(product_name)

# Define a prompt template
prompt_template = ChatPromptTemplate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Given the context: {context}, answer the question: {question}"}
    ]
)

# Function to generate response
def generate_response(product_name):
    # Retrieve context using the product name
    context = retrieve_sales_data(product_name)

    # Test with a prompt
    input_text = f"What was the total sales amount for {product_name} in the last month?"
    prompt = prompt_template.format(context=context, question=input_text)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
        
    # Decode response
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    return response_text

# Example usage
if __name__ == "__main__":
    response = generate_response(product_name)
    print(response)