import os
from pymongo import MongoClient
from dotenv import load_dotenv


# Get the backend folder path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Explicitly load backend/.env
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)


MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")


print("====================================")
print("MongoDB URI:", MONGO_URI)
print("Database Name:", DATABASE_NAME)
print("====================================")


if not MONGO_URI:
    raise Exception("MONGO_URI is not found in .env file")

if not DATABASE_NAME:
    raise Exception("DATABASE_NAME is not found in .env file")


client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=1500,
    connectTimeoutMS=1500
)

db = client[DATABASE_NAME]

conversations_collection = db["conversations"]
sessions_collection = db["sessions"]
candidates_collection = db["candidates"]
user_memory_collection = db["user_memory"]
documents_collection = db["uploaded_documents"]

# Test connection
try:
    client.admin.command("ping")

    print("MongoDB connected successfully!")
    print("Database:", db.name)
    print("Collection (sessions):", sessions_collection.name)
    print("Collection (user_memory):", user_memory_collection.name)
    print("Collection (uploaded_documents):", documents_collection.name)
    print("Collection (candidates):", candidates_collection.name)

except Exception as e:
    print("MongoDB connection failed (running in local offline fallback):", e)