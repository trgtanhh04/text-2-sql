import psycopg2, csv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATABASE_URL

def connect_db():
    conn = psycopg2.connect(DATABASE_URL)
    return conn