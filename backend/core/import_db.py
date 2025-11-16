import psycopg2, csv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATABASE_URL


conn = psycopg2.connect(DATABASE_URL)

cur = conn.cursor()

with open("../data/sales_data.csv", "r", encoding="utf8") as f:
    next(f)
    cur.copy_from(f, "sales_data", sep=",")

cur = conn.cursor()

with open("../data/sales_data.csv", "r", encoding="utf8") as f:
    next(f)
    cur.copy_from(f, "sales_data", sep=",")

conn.commit()
cur.close()
conn.close()

print("Imported successfully!")
