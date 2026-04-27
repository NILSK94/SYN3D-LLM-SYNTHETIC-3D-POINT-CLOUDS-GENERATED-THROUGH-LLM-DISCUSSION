import os
import threading
from typing import Optional
from openai import OpenAI

# ============================================================
# Config definitions
# ============================================================

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Directories
ROOT_OUT = os.getenv("SYN3D_OUT_DIR", os.path.join(os.getcwd(), "syn3d_runs"))
DATA_DIR = os.path.join(ROOT_OUT, "data")
LOG_DIR = os.path.join(ROOT_OUT, "logs")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Global State for API
_client: Optional[OpenAI] = None
_current_api_key: str = DEFAULT_API_KEY
_api_lock = threading.Lock()

def get_client() -> Optional[OpenAI]:
    with _api_lock:
        return _client

def set_openai_client(api_key: str):
    global _client, _current_api_key
    api_key = (api_key or "").strip()
    with _api_lock:
        _current_api_key = api_key
        if not api_key:
            _client = None
            return
        _client = OpenAI(api_key=api_key)

# Initialize if we have a default key
if DEFAULT_API_KEY:
    set_openai_client(DEFAULT_API_KEY)
