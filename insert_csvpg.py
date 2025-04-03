import pandas as pd
from sqlalchemy import create_engine, text

df = pd.read_csv('april2.csv')
df.columns = [col.lower() for col in df.columns]

engine = create_engine('postgresql://postgres:cicada3301@localhost:5432/sales_data')

with engine.connect() as connection:
    columns = connection.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'sales_data'"))
    existing_columns = [col[0] for col in columns]
    print("Existing columns:", existing_columns)

df.to_sql('sales_data', engine, if_exists='append', index=False)