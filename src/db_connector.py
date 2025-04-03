import mysql.connector
from langchain_community.utilities import SQLDatabase


def get_mysql_connection():
    return mysql.connector.connect(host="localhost", user="root", password="", database="sales_data")


def get_database_schema():
    db = SQLDatabase.from_uri("mysql+pymysql://root:@localhost/sales_data")
    tables = db.get_usable_table_names()
    schema_info = "\n\n".join([db.get_table_info(table) for table in tables])
    with open("./data/schema_info.txt", "w") as f:
        f.write(schema_info)
    return schema_info

def fetch_sales_data():
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT orderDate, storeName, productName, quantity, totalProductPrice FROM sales_data LIMIT 500")
    data = cursor.fetchall()
    conn.close()
    return data


def get_column_names():
    """Fetch column names dynamically from the sales_data table."""
    conn = get_mysql_connection() 
    cursor = conn.cursor()
    cursor.execute("SHOW COLUMNS FROM sales_data") 
    columns = [row[0] for row in cursor.fetchall()]
    conn.close()
    return columns
