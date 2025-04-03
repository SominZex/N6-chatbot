
import os
import re
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sqlparse
import pandas as pd
import plotly.express as px
import streamlit as st
from functools import lru_cache
from langchain_groq import ChatGroq
import psycopg2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sql_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "cicada3301")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "sales_data")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Extract database schema
@lru_cache(maxsize=1)
def get_database_schema():
    inspector = inspect(engine)
    schema = ""
    for table_name in inspector.get_table_names():
        if table_name == "sales_data":
            schema += f"Table: {table_name}\n"
            for column in inspector.get_columns(table_name):
                schema += f"- {column['name']}: {column['type']}\n"
            schema += "\n"
    logger.info("Retrieved database schema.")
    return schema

# Fetch dynamic values from database
@lru_cache(maxsize=1)
def get_dynamic_values():
    session = SessionLocal()
    try:
        brands_query = text("SELECT DISTINCT brandName FROM sales_data WHERE brandName IS NOT NULL AND brandName != ''")
        brands_result = session.execute(brands_query)
        brands = [row[0] for row in brands_result.fetchall()]
        
        categories_query = text("SELECT DISTINCT categoryName FROM sales_data WHERE categoryName IS NOT NULL AND categoryName != ''")
        categories_result = session.execute(categories_query)
        categories = [row[0] for row in categories_result.fetchall()]
        
        stores_query = text("SELECT DISTINCT storeName FROM sales_data WHERE storeName IS NOT NULL AND storeName != ''")
        stores_result = session.execute(stores_query)
        stores = [row[0] for row in stores_result.fetchall()]
        
        products_query = text("SELECT DISTINCT productName FROM sales_data WHERE productName IS NOT NULL AND productName != '' LIMIT 1000")
        products_result = session.execute(products_query)
        products = [row[0] for row in products_result.fetchall()]
        
        payment_methods_query = text("SELECT DISTINCT paymentMethod FROM sales_data WHERE paymentMethod IS NOT NULL AND paymentMethod != ''")
        payment_methods_result = session.execute(payment_methods_query)
        payment_methods = [row[0] for row in payment_methods_result.fetchall()]
        
        order_types_query = text("SELECT DISTINCT orderType FROM sales_data WHERE orderType IS NOT NULL AND orderType != ''")
        order_types_result = session.execute(order_types_query)
        order_types = [row[0] for row in order_types_result.fetchall()]
        
        order_from_query = text("SELECT DISTINCT orderFrom FROM sales_data WHERE orderFrom IS NOT NULL AND orderFrom != ''")
        order_from_result = session.execute(order_from_query)
        order_from = [row[0] for row in order_from_result.fetchall()]
        
        return {
            "brands": brands,
            "categories": categories,
            "stores": stores,
            "products": products,
            "payment_methods": payment_methods,
            "order_types": order_types,
            "order_from": order_from
        }
    except Exception as e:
        logger.error(f"Error fetching dynamic values: {str(e)}")
        return {
            "brands": [],
            "categories": [],
            "stores": [],
            "products": [],
            "payment_methods": [],
            "order_types": [],
            "order_from": []
        }
    finally:
        session.close()

# casual intent detector
def is_general_conversation(question: str):
    casual_keywords = [
        "hi", "hello", "how are you", "what's up", "hey",
        "good morning", "good evening", "ok", "okay",
        "thanks", "thank you", "cool", "great", "awesome", "sure", "nice", "sounds good"
    ]
    normalized = question.lower().strip()
    
    if normalized in casual_keywords:
        return True
    
    if len(normalized.split()) <= 10 and any(normalized.startswith(kw) for kw in casual_keywords):
        return True

    return False

llm=ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1)

# General conversation
def handle_general_chat(question: str):
    system = "You are a helpful AI assistant capable of normal conversation."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}")
    ])
    general_response = prompt | llm | StrOutputParser()
    response = general_response.invoke({"question": question})
    return response

def extract_sql(raw_output: str):

    response = raw_output.strip()
    response = re.sub(r"(?i)sql query:|```sql|```|^sql\\s*:", "", response)
    sql_code = re.search(r"(SELECT|WITH)[\\s\\S]+", response, re.IGNORECASE)

    if sql_code:
        sql = sql_code.group(0).strip()
        if "FROM (" in sql and "subquery" in sql.lower():
            if "AVG(totalProductPrice)" in sql:
                sql = sql.replace("AVG(totalProductPrice)", "AVG(total_sales)")
            if "SUM(totalProductPrice)" in sql and "GROUP BY storeName" in sql:
                sql = sql.replace("SUM(totalProductPrice)", "SUM(totalProductPrice) AS total_sales")
        return sqlparse.format(sql, reindent=True, keyword_case='upper')

    agg_pattern = r'SELECT\s+[^;]*SUM\([^;]*\)[^;]*;'
    agg_match = re.search(agg_pattern, raw_output, re.IGNORECASE | re.DOTALL)
    if agg_match:
        return agg_match.group(0).strip()
    
    code_blocks = re.findall(r'```sql(.*?)```', raw_output, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    any_code_blocks = re.findall(r'```(.*?)```', raw_output, re.DOTALL)
    if any_code_blocks:
        for block in any_code_blocks:
            if "SELECT" in block.upper():
                return block.strip()
    
    select_match = re.search(r'SELECT[\s\S]*?;', raw_output, re.IGNORECASE)
    if select_match:
        sql = select_match.group(0).strip()
        
        if ('sales' in raw_output.lower() or 'brand' in raw_output.lower()) and 'SUM(' not in sql:
            if 'brandName' in sql and 'totalProductPrice' in sql:
                table = 'sales_data'
                where_clause = re.search(r'WHERE\s+(.*?)(GROUP BY|ORDER BY|LIMIT|$)', sql + ' ', re.IGNORECASE)
                conditions = where_clause.group(1) if where_clause else ""
                
                return f"SELECT brandName, SUM(totalProductPrice) AS total_sales FROM {table} WHERE {conditions.strip()} GROUP BY brandName ORDER BY total_sales DESC;"
        
        return sql
    
    return ""

# For business or data-related question
def is_business_query(question: str):

    schema = get_database_schema()

    column_pattern = r'- (\w+):'
    business_keywords = re.findall(column_pattern, schema)
    
    # business terms
    business_keywords.extend(["sales", "orders", "amount", "quantity", "product", "store", "brand", 
                            "profit", "loss", "customer", "gst", "payment", "category"])
    
    follow_ups = ["no, i mean", "actually", "what i meant", "let me rephrase", "no i mean", "i mean"]
    normalized = question.lower()
    
    for follow in follow_ups:
        if follow in normalized:
            return True
            
    date_indicators = ["yesterday", "today", "march", "april", "may", "june", "july", "august", 
                      "september", "october", "november", "december", "january", "february", 
                      "2024", "2025"]
    
    if any(date in normalized for date in date_indicators):
        return True
        
    return any(kw.lower() in normalized for kw in business_keywords)


def convert_nl_to_sql(question: str):
    today = datetime.today()
    yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    today_str = today.strftime('%Y-%m-%d')
    current_year = today.year

    normalized_question = question.lower()

    date_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})'
    date_matches = re.findall(date_pattern, normalized_question)
    
    mentioned_date = None
    if date_matches:
        month_name, day, year = date_matches[0]
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        month = month_map.get(month_name)
        if month:
            try:
                mentioned_date = f"{year}-{month:02d}-{int(day):02d}"
            except ValueError:
                pass
    
    follow_ups = ["no, i mean", "actually", "what i meant", "let me rephrase", "no i mean", "i mean"]
    cleaned_question = question
    for follow in follow_ups:
        if follow in normalized_question:
            cleaned_question = question.lower().replace(follow, "").strip()
            break

    dynamic_values = get_dynamic_values()
    
    schema = get_database_schema()
    example_prompts = """Example Prompts:
    1. What is the total sales on {{orderDate}}?
    2. How many orders were placed on {{orderDate}}?
    3. What is the total sales for {{storeName}} on {{orderDate}}?
    4. What is the average orderAmountNet for {{month/year}}?
    5. Show me the totalProductPrice for {{productName}} on {{orderDate}}.
    6. What is the total GSTAmount collected on {{orderDate}}?
    7. How much was paid via cardAmount on {{orderDate}}?
    8. What is the total quantity sold for {{productName}} in {{month/year}}?
    9. What is the total sales grouped by brandName in {{month/year}}?
    10. What is the total orderAmountNet by subCategoryOf in {{storeName}}?
    11. What is the total sales where GSTIN is {{GSTIN}}?
    12. Show me total sales for paymentMethod {{paymentMethod}} in {{storeName}}.
    13. List total sales where customerName is {{customerName}}.
    14. What is the total sales by orderType {{orderType}}?
    15. Show sales data for orders from {{orderFrom}} platform.
    16. List total sales grouped by categoryName.
    17. What is the sales for brandName {{brandName}} on {{orderDate}}?
    18. What is the median daily sales for stores?
    19. What is the standard deviation of total sales across stores?
    20. What is the 90th percentile of daily sales per store?
    21. What is the variance of daily sales for each category?
    22. Calculate the quartiles (Q1, Q2, Q3) for total sales grouped by storeName.
    23. What is the cumulative sum of sales grouped by orderDate?
    24. Calculate the growth rate of sales month over month.
    25. What is the moving average of daily sales with a 7-day window?
    26. What is the correlation between quantity and totalProductPrice?
    27. What is the maximum and minimum sales per brandName?
    28. What is the ratio of cashAmount to cardAmount by store?
    29. What are the total sales of each stores yesterday?
    30. What are the {{productName}} available in {{brandName}} brand?
    31. What are the {{brandName}} available in the {{categoryName}} category?
    32. What are the top selling {{brandName}} brand?
    33. What are the top selling {{productName}} ?
    34. What are the top selling {{brandName}} brand?
    35. What are the top selling {{productName}}?
    36. List all products under brandName {{brandName}}?
    37. What are the total sales for each product {{productName}} under brand {{brandName}}?
    38. Plot a line graph of daily sales for last 7 days.
    39. Plot a bar graph or line graph for trend analysis of sales only of top 20.
    40. Show me a bar chart of total sales for top 20 stores.
    41. Show me a line chart, bar chart or any meaningful chart for top 20 sales data we have. 
    42. Show me the chart of top 20 product{{productName}} sales.
    43. Show me the chart of top 20 brand {{brandName}} sales.
    44. Show me the chart of top 20 category{{categoryName}} sales.
    45. Show me the chart of top 20 product{{productName}} sales of last week.
    46. Show me the chart of top 20 brand {{brandName}} sales of last week.
    47. Show me the chart of top 20 brand {{brandName}} sales of last month.
    48. Show me the chart of top 20 category {{categoryName}} sales of last week.
    49. Tell me sales of {{brandName}} brand on {{orderDate}} also give me the chart.
    50. What is sales of products of {{brandName}} on yesterday?
    51. What is sales of products of {{brandName}} on {{orderDate}}?
    52. What is sales of {{brandName}} products on {{orderDate}}?
    52. What is sales of {{brandName}} products on yesterday?
    """
    system_prompt = f"""You are an assistant that converts natural language questions into SQL queries.

Today's date is {today_str} and yesterday's date is {yesterday}.

Schema:
{schema}

{example_prompts}

Guidelines:
- ONLY use tables and columns provided in the schema.
- NEVER use JOIN operations under any circumstances.
- ONLY query FROM the sales_data table.
- For 'sales', always use SUM(totalProductPrice) AS total_sales.
- Use correct date formats: WHERE orderDate = 'YYYY-MM-DD'.
- If the user mentions 'yesterday', convert it to: WHERE orderDate = '{yesterday}'.
- If the user mentions 'today', convert it to: WHERE orderDate = '{today_str}'.
- If the user mentions a month and day without a year, default to the current year {current_year}.
- When filtering storeName, productName, brandName, categoryName, subCategoryOf, customerName, GSTIN, and orderFrom use LIKE '%value%' to allow for spelling variations.
- If the user asks for a specific date like "February 28, 2025", always convert it to: WHERE orderDate = '2025-02-28'.
- AVOID using nested JOINs and CASE WHEN unless strictly necessary.
- NEVER use MONTH() or YEAR() functions unless explicitly asked for month-level or year-level aggregation.
- For calculating median, explain that PostgreSQL requires custom logic such as ordering by totalProductPrice, LIMIT + OFFSET.
- Always prefer simple aggregate queries when possible.
- Always sort the sales in descending (DESC) order for any queries.
- Don't use storeInvoice column when questions like total sales of each store is asked, instead use storeName and include the storeName column to make it easier to understand also sort the sales in descending order.
- If the prompt or the question is about sales then always use Indian currency standard. Don't use any other currency.
- Always sort any sales in descending order.
- Show the charts only when asked by the user.
- Always use correct columns for GROUP BY and ORDER BY based on the context.
- When the question is about brand, category and store you have use brandName, categoryName and storeName respectively.
- Don't get confused when the question is about brand sales. You should simply use brand sales and use as simple SQL query as possible. Never use additional columns like storeName unless asked. 
- Always ORDER BY total_sales DESC and LIMIT to top 20 if user asks for "top brands", "top stores", or "top categories".
- Do NOT include extra columns like storeName when the user only asks for brand sales.
- If the user asks for only "brand sales", NEVER include storeName or any other grouping besides brandName.
- If the user asks for only "category sales", NEVER include storeName or any other grouping besides categoryName.
- If the user asks for only "store sales", NEVER include storeName or any other grouping besides storeName.
- If the user asks for brand-level data, DO NOT include storeName unless it is part of the question.
- When the user mentions a brand and a specific date, filter directly on brandName and orderDate avoid using storeName unless mentioned.
- For aggregation (e.g., SUM, COUNT, AVG), just aggregate at the mentioned level (brandName, categoryName, etc.).
- Avoid unnecessary subqueries unless needed for logic like "CASE WHEN".
- DO NOT include 'storeName' in SELECT, GROUP BY, or WHERE unless the question mentions "store" or "stores".
- If the user asks for brand sales, only filter by 'brandName'.
- Only use GROUP BY if the user explicitly requests a breakdown (e.g., "by store", "by brand").
- Aggregate sales using SUM(totalProductPrice) unless specified otherwise.
- For questions asking for data on a single day, use WHERE orderDate = '<date>'.
- Avoid unnecessary columns. Only include what the question asks for.
- Keep the query clean and minimal.

Special Instruction:


- To calculate the median stores, use Indian currency standard and use the following logic:
SELECT 
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_sales) AS median_sales
FROM (
    SELECT storeName, SUM(totalProductPrice) AS total_sales
    FROM sales_data
    GROUP BY storeName
) AS sub;


- For **percentiles** or **quartiles**, use:
  SELECT storeName, PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY totalProductPrice) AS Q1, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY totalProductPrice) AS Median, PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY totalProductPrice) AS Q3 FROM sales_data GROUP BY storeName;

- For **variance** and **stddev**:
  SELECT storeName, VARIANCE(totalProductPrice) AS variance_sales, STDDEV(totalProductPrice) AS stddev_sales FROM sales_data GROUP BY storeName;

- For **cumulative sum**:
  SELECT orderDate, storeName, SUM(totalProductPrice) OVER (PARTITION BY storeName ORDER BY orderDate ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_sales FROM sales_data;

- For query what is the total sales of all brands yeserday: 
  SELECT brandName, SUM(totalProductPrice) AS total_sales FROM sales_data WHERE orderDate = '{yesterday}' GROUP BY brandName ORDER BY total_sales DESC LIMIT 20;

- For query what is the sales of all stores yesterday:
  SELECT storeName, SUM(totalProductPrice) AS total_sales FROM sales_data WHERE orderDate = '{yesterday}' GROUP BY storeName ORDER BY total_sales DESC LIMIT 20;

- For query what is the sales of all categories yesetday:
  SELECT categoryName, SUM(totalProductPrice) AS total_sales FROM sales_data WHERE orderDate = '{yesterday}' GROUP BY categoryName ORDER BY total_sales DESC LIMIT 20;

- For ANY question about sales, ALWAYS use SUM(totalProductPrice) AS total_sales.

- When user asks "What is sales of X brand on <date>" ALWAYS use this template:
  SELECT brandName, SUM(totalProductPrice) AS total_sales 
  FROM sales_data 
  WHERE brandName LIKE '%X%' AND orderDate = 'YYYY-MM-DD';

- When user asks "What is sales of X brand on yesterday" ALWAYS use this template:
  SELECT brandName, SUM(totalProductPrice) AS total_sales 
  FROM sales_data 
  WHERE brandName LIKE '%X%' AND orderDate = '{yesterday}';

- When user asks "What is sales of X Category on <date>" ALWAYS use this template:
  SELECT brandName, SUM(totalProductPrice) AS total_sales 
  FROM sales_data 
  WHERE categoryName LIKE '%X%' AND orderDate = 'YYYY-MM-DD';

- When user asks "What is sales of X Category on yesterday" ALWAYS use this template:
  SELECT brandName, SUM(totalProductPrice) AS total_sales 
  FROM sales_data 
  WHERE categoryName LIKE '%X%' AND orderDate = '{yesterday}';

- When user asks "What is sales of X store on <date>" ALWAYS use this template:
  SELECT brandName, SUM(totalProductPrice) AS total_sales 
  FROM sales_data 
  WHERE storeName LIKE '%X%' AND orderDate = 'YYYY-MM-DD';

- When user asks "What is sales of X store on yesterday" ALWAYS use this template:
  SELECT brandName, SUM(totalProductPrice) AS total_sales 
  FROM sales_data 
  WHERE storeName LIKE '%X%' AND orderDate = '{yesterday}';

- When user asks "What is sales of X product on <date>" ALWAYS use this template:
  SELECT brandName, SUM(totalProductPrice) AS total_sales 
  FROM sales_data 
  WHERE productName LIKE '%X%' AND orderDate = 'YYYY-MM-DD';

- When user asks "What is sales of X product on yesterday" ALWAYS use this template:
  SELECT brandName, SUM(totalProductPrice) AS total_sales 
  FROM sales_data 
  WHERE productName LIKE '%X%' AND orderDate = '{yesterday}';

- When user asks "What is sales of X product on <date>" ALWAYS use this template:
  SELECT brandName, SUM(totalProductPrice) AS total_sales 
  FROM sales_data 
  WHERE productName LIKE '%X%' AND orderDate = 'YYYY-MM-DD';

- When user asks "What is sales of X product on yesterday" ALWAYS use this template:
  SELECT brandName, SUM(totalProductPrice) AS total_sales 
  FROM sales_data 
  WHERE productName LIKE '%X%' AND orderDate = '{yesterday}';

- When user asks "What is sales of products of X brand on <date>" or "What is the sales of X brand products on <date>?" ALWAYS use this template:
    SELECT productName, sum(totalProductPrice) as sales from sales_data
    where brandName LIKE 'X%' and orderDate = "YYYY-MM-DD"
    group by productName
    order by sales desc;

- When user asks "What is sales of products of X brand on yesterday" or "What is sales of X brand products on yesterday?" ALWAYS use this template:
    SELECT productName, sum(totalProductPrice) as sales from sales_data
    where brandName LIKE 'X%' and orderDate = "{yesterday}"
    group by productName
    order by sales desc;



Question: {question}
"""

    convert_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    sql_generator = convert_prompt | llm | StrOutputParser()
    raw_output = sql_generator.invoke({"question": question})
    cleaned_query = extract_sql(raw_output)

    logger.info(f"Generated SQL query: {cleaned_query}")
    return cleaned_query



# Execute SQL
def execute_sql(sql_query: str):
    if not sql_query.lower().startswith("select"):
        logger.error("Detected non-SQL or malformed query")
        return pd.DataFrame()

    session = SessionLocal()
    logger.info(f"Executing SQL query: {sql_query}")
    try:
        result = session.execute(text(sql_query))
        rows = result.fetchall()
        columns = result.keys()
        data = pd.DataFrame(rows, columns=columns)
        logger.info(f"SQL Query returned {len(data)} rows")
        return data
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def generate_chart(question: str, df: pd.DataFrame):
    if 'chart' in question.lower() or 'graph' in question.lower() or 'plot' in question.lower():
        columns = df.columns.tolist()

        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = [col for col in columns if col not in numeric_columns]
        
        x_axis_candidates = [col for col in categorical_columns if col != 'total_sales']
        if not x_axis_candidates:
            x_axis_candidates = categorical_columns
        
        y_axis_candidates = [col for col in numeric_columns if 'total' in col.lower() or 'sum' in col.lower() or 
                             'sales' in col.lower() or 'amount' in col.lower() or 'price' in col.lower() or 
                             'quantity' in col.lower()]
        if not y_axis_candidates:
            y_axis_candidates = numeric_columns

        x_col = x_axis_candidates[0] if x_axis_candidates else columns[0]
        y_col = y_axis_candidates[0] if y_axis_candidates else columns[1] if len(columns) > 1 else None

        if any(key in question.lower() for key in ['orderstatus', 'paymentmethod', 'ordertype']) and len(columns) >= 2:
            fig = px.pie(df, names=columns[0], values=columns[1], hole=0.5,
                         title=f"{columns[0]} Distribution", template='plotly_dark')
        elif 'daily' in question.lower() and 'orderDate' in columns and y_col:
            fig = px.line(df, x='orderDate', y=y_col, markers=True,
                          title='Daily Trend', template='plotly_dark')
        elif y_col:
            fig = px.bar(df, x=x_col, y=y_col, color=y_col,
                         title=f'{x_col} vs {y_col}', template='plotly_dark')
        else:
            return None

        return fig

    return None

# Convert SQL results to natural language
def generate_human_readable_answer(question: str, sql_query: str, query_rows: pd.DataFrame):

    system = """You are an assistant that interprets SQL query results into clear, concise business insights.
    - Always assume that the provided data is pre-sorted if an ORDER BY clause was present in the SQL.
    - Store sale, Product sale, Brand sale and Category sale should always be ranked in Descending order.
    - Do NOT infer additional rankings beyond what is provided.
    - You should be smart enough to understand business questions.
    - When total_sales is None or the dataset is empty, DO NOT say there are no sales. Instead say: "Based on the available data, there were no recorded sales for the specified criteria."
    - Always use India currency standard (INR) wherever necessary instead of Dollar or any other currency format.
    - Make sure your answer is directly tied to the query result.
    - If the data is empty or contains null values, provide a user-friendly explanation about what that means.
    """

    if query_rows.empty or (len(query_rows) == 1 and query_rows.iloc[0].isnull().all()):
        return "Based on the available data, there were no recorded sales for the specified criteria."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}\nSQL Query: {sql}\nData: {data}\nProvide a business-friendly answer.")
    ])

    human_response = prompt | llm | StrOutputParser()
    response = human_response.invoke({"question": question, "sql": sql_query, "data": query_rows.to_dict(orient='records')})
    logger.info("Generated human-readable answer.")
    return response

# Query pipeline
def query_handler(question: str):
    logger.info(f"Received question: {question}")

    if is_general_conversation(question) and not is_business_query(question):
        return handle_general_chat(question), None

    sql_query = convert_nl_to_sql(question)
    
    if not sql_query:

        logger.warning("LLM returned empty SQL, providing fallback response.")
        
        dynamic_values = get_dynamic_values()
        normalized = question.lower()
        
        brand_mentions = [brand for brand in dynamic_values["brands"] if brand.lower() in normalized]
        store_mentions = [store for store in dynamic_values["stores"] if store.lower() in normalized]
        date_mentions = any(date in normalized for date in ["yesterday", "today", "march", "april"])
        
        if brand_mentions:
            return f"I understand you're asking about {brand_mentions[0]} sales, but I'm having trouble formulating the right query. Could you please specify which time period you're interested in? For example, 'What were {brand_mentions[0]} sales yesterday?' or 'What were {brand_mentions[0]} sales on March 20, 2025?'", None
        elif store_mentions:
            return f"I understand you're asking about {store_mentions[0]} store, but I'm having trouble formulating the right query. Could you please specify what sales information you're looking for? For example, 'What were total sales at {store_mentions[0]} yesterday?'", None
        elif date_mentions:
            return "I see you're asking about a specific date. Could you please clarify what sales information you're looking for on that date? For example, 'What were total sales yesterday?' or 'What were the sales for a specific brand yesterday?'", None
        else:
            return "I'm not able to generate a SQL query for that question. Could you please rephrase your question or provide more specific details about what sales information you're looking for?", None

    try:
        df = execute_sql(sql_query)
        chart = generate_chart(question, df) if not df.empty else None
        response = generate_human_readable_answer(question, sql_query, df)
        return response, chart
    except Exception as e:
        logger.error(f"Error in query execution: {str(e)}")
        return f"I encountered an error while processing your question. Please try rephrasing it or check if the data you're asking about exists.", None


if __name__ == "__main__":
    question = input("Insert Prompt: ")
    result = query_handler(question)
    print(result)
