import sqlite3
import pandas as pd

conn = sqlite3.connect("sp500_data.db")
cur = conn.cursor()

# list tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print("Tables found:", tables)

# peek at the first table automatically
if tables:
    table_name = tables[0][0]
    print(f"\nPreviewing table: {table_name}\n")
    df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10", conn)
    print(df)
    print("\nColumns:", list(df.columns))