import pymysql

def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="sales_data",
        cursorclass=pymysql.cursors.DictCursor
    )
