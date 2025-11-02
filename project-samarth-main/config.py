# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (local dev). On Streamlit Cloud, use Secrets.
load_dotenv()

# --- Base paths anchored to this file (works when nested under project-samarth-main/) ---
BASE_DIR = Path(__file__).resolve().parent

# Data paths
RAW_DATA_DIR = str((BASE_DIR / "data" / "raw").resolve())
PROCESSED_DATA_DIR = str((BASE_DIR / "data" / "processed").resolve())

# ChromaDB (prebuilt, persisted) — default to ./chroma_db next to app/config
DEFAULT_CHROMA = str((BASE_DIR / "chroma_db").resolve())
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", DEFAULT_CHROMA)

# Collection name MUST match the one used to build the index
COLLECTION_NAME = "agricultural_data"

# API keys (optional; use Streamlit Secrets in prod)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATAGOVINDIA_API_KEY = os.getenv("DATAGOVINDIA_API_KEY", "")
BASE_API_URL = "https://api.data.gov.in/resource"
CROP_PRODUCTION_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

# Logging (optional)
LOG_FILE = str((BASE_DIR / "logs" / "app.log"))



