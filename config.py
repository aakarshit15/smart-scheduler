import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GROQ_MODEL = os.getenv("FAST_LLM")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")
if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY not found in .env file")