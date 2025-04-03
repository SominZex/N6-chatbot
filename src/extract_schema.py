import mysql.connector


conn = mysql.connector.connect(
    host="localhost",
    user="root", 
    password="",  
    database="sales_data"  
)

cursor = conn.cursor()


cursor.execute("SHOW TABLES;")
tables = [table[0] for table in cursor.fetchall()]

schema_info = ""

for table in tables:
    cursor.execute(f"DESCRIBE {table};")
    columns = cursor.fetchall()
    
    schema_info += f"Table: {table}\n"
    schema_info += "\n".join([f"  - {col[0]} ({col[1]})" for col in columns])
    schema_info += "\n\n"

with open("./data/schema_info.txt", "w") as f:
    f.write(schema_info)

print("Database schema saved to `data/schema_info.txt`")

# Close connection
cursor.close()
conn.close()
