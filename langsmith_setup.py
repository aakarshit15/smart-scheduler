# langsmith_setup.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import Client

load_dotenv()

print("Checking LangSmith Configuration...")
print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")
print(f"LANGSMITH_API_KEY: {'Set' if os.getenv('LANGSMITH_API_KEY') else 'Missing'}")

try:
    client = Client()
    print(f"LangSmith Connected: {client.api_url}")
    
    try:
        projects = list(client.list_projects(limit=5))
        print(f"Found {len(projects)} project(s)")
    except Exception as e:
        print(f"Project list: {e}")
        
except Exception as e:
    print(f"LangSmith Connection Error: {e}")
    print("Make sure LANGSMITH_API_KEY is set in .env file")
    exit(1)

print("Testing LangSmith Tracing...")
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    response = llm.invoke("What is 2+2? Answer in one word.")
    
    print(f"LLM Response: {response.content}")
    
except Exception as e:
    print(f"LLM Test Error: {e}")
