import pymysql
from sqlalchemy import create_engine

def get_db_connection():
    """Create and return a MySQL database connection using XAMPP."""
    username = "root"  
    password = ""  
    host = "127.0.0.1"
    database = "sales_data"

    engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database}")
    return engine

