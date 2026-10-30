
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from project root
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "neuralshield")

if not MONGODB_URI or "YOUR_NEW_" in MONGODB_URI:
    raise ValueError(
        "MongoDB URI missing. Check your .env file."
    )

# Create MongoDB connection
client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=10000
)

# Select database
db = client[DATABASE_NAME]

# Collections
detections_collection = db["detections"]
alerts_collection = db["alerts"]
sessions_collection = db["monitor_sessions"]
model_versions_collection = db["model_versions"]
audit_logs_collection = db["audit_logs"]


def test_connection():

    try:
        client.admin.command("ping")

        print("MongoDB connection successful!")
        print("Database:", DATABASE_NAME)

        return True

    except Exception as error:

        print("MongoDB connection failed!")
        print("Error:", error)

        return False
