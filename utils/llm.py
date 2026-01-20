import os
import sys
import config
from dotenv import load_dotenv
from langsmith import Client
from langchain_groq import ChatGroq


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "smart-scheduler-new"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

try:
    client = Client()
    print("LangSmith Client initialized successfully")
except Exception as e:
    print(f"LangSmith setup error: {e}")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))



def get_llm(temperature=0, model=None):
    if model is None:
        model = config.GROQ_MODEL
        
    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=config.GROQ_API_KEY,
    )