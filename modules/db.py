# modules/db.py
"""
MongoDB Connection Module
- Single source of truth for all collections
- TLS-enabled Atlas connection
- No duplicate assignments
"""

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get connection details
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

# Validate required env vars
if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found in environment variables")
if not DB_NAME:
    raise ValueError("DB_NAME not found in environment variables")

# Connect to MongoDB Atlas with TLS
try:
    client = MongoClient(
        MONGODB_URI,
        tls=True,
        tlsAllowInvalidCertificates=False,  # Set True only for self-signed certs
        serverSelectionTimeoutMS=5000
    )
    client.admin.command("ping")
    print("Connected to MongoDB Atlas successfully!")
except ServerSelectionTimeoutError as e:
    raise ConnectionError(f"Could not connect to MongoDB Atlas: {e}") from e

# Database and collections (single, clean assignment)
db = client[DB_NAME]

users_collection = db["users"]
user_profiles = db["user_profiles"]
expenses_collection = db["expenses"]
debt_collection = db["debt"]
investment_collection = db["investments"]
carloans_collection = db["carLoans"]
mortgage_collection = db["mortgage_calculations"]
insurance_collection = db["insurance"]