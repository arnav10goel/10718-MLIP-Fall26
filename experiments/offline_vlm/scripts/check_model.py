"""Exit 0 if the requested Gemini model is available to the key in .env. Never prints the key."""

import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from google import genai

REPO = Path(__file__).resolve().parents[3]
wanted = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.5-flash-lite"
key = os.environ.get("GEMINI_API_KEY") or dotenv_values(REPO / ".env").get("GEMINI_API_KEY")
if not key:
    sys.exit("no GEMINI_API_KEY")
client = genai.Client(api_key=key)
try:
    names = [m.name.removeprefix("models/") for m in client.models.list()]
except Exception as error:  # do not print the message: it could echo the key
    sys.exit(f"model list failed: {type(error).__name__}")
print("models with 3.5:", sorted(n for n in names if "3.5" in n))
sys.exit(0 if wanted in names else f"{wanted} not available")
