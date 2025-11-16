import os
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+pysqlite:///./test.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Neon DB connection parameters
HOST = os.getenv("HOST", "")
DBNAME = os.getenv("DBNAME", "")
USER = os.getenv("USER", "")
PASSWORD = os.getenv("PASSWORD", "")
SSLMODE = os.getenv("SSLMODE", "require")
DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://{USER}:{PASSWORD}@{HOST}/{DBNAME}?sslmode={SSLMODE}&channel_binding=require")

print("Configuration loaded:")
print(f"DATABASE_URL: {DATABASE_URL}")
