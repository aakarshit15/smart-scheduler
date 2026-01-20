import os
from dotenv import load_dotenv
from langsmith import Client

def setup_tracing():
    load_dotenv()
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "smart-scheduler"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    
    try:
        client = Client()
        print("LangSmith tracing initialized")
        return True
    except Exception as e:
        print(f"LangSmith error: {e}")
        return False

setup_tracing()
