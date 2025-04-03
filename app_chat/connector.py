from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from langchain.sql_database import SQLDatabase

sales_connection_url = URL.create(
 "mssql+pymssql",
 username = "root",
 password = "",
 host = "127.0.0.1",
 database = sales_data )
sales_db_engine = SQLDatabase(create_engine(sales_connection_url))