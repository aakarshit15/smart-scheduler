import os
from dotenv import load_dotenv
import config
from utils.tracing import setup_tracing
from langsmith import Client
import sys
from typing import Dict, List
from agents.state import SchedulerState, Task
from utils.document_processor import DocumentProcessor
from utils.llm import get_llm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "smart-scheduler-new"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

try:
    client = Client()
except:
    pass






class TaskExtractorAgent:
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm = get_llm(temperature=0)
    
    def process(self, state: SchedulerState) -> Dict:
        state["messages"].append("Task Extractor Agent: Starting extraction...")
        state["current_step"] = "extract_tasks"
        
        all_text = ""
        
        if state.get("raw_input"):
            all_text += state["raw_input"] + "\n\n"
            state["messages"].append(f"Processing text input ({len(state['raw_input'])} chars)")
        
        if state.get("uploaded_file_paths"):
            file_paths = state["uploaded_file_paths"]
            state["messages"].append(f"Processing {len(file_paths)} uploaded file(s)...")
            
            for idx, file_path in enumerate(file_paths, 1):
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    
                    if file_path.lower().endswith('.pdf'):
                        state["messages"].append(f"[{idx}/{len(file_paths)}] Processing PDF: {file_name}...")
                        pdf_text = self.doc_processor.process_pdf(file_path)
                        all_text += f"\n\n--- Content from {file_name} ---\n\n" + pdf_text + "\n\n"
                        
                    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        state["messages"].append(f"[{idx}/{len(file_paths)}] Processing image: {file_name}...")
                        image_text = self.doc_processor.process_image_with_llm(file_path)
                        all_text += f"\n\n--- Content from {file_name} ---\n\n" + image_text + "\n\n"
                        
                    elif file_path.lower().endswith('.txt'):
                        state["messages"].append(f"[{idx}/{len(file_paths)}] Processing text file: {file_name}...")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            all_text += f"\n\n--- Content from {file_name} ---\n\n" + f.read() + "\n\n"
                            
                    else:
                        state["messages"].append(f"[{idx}/{len(file_paths)}] Unsupported file type: {file_name}")
                else:
                    state["messages"].append(f"File not found: {file_path}")

        
        if all_text.strip():
            state["messages"].append(f"Analyzing {len(all_text)} characters of content...")
            tasks = self.doc_processor.extract_tasks_from_text(all_text)
            
            extracted_tasks = []
            for task_dict in tasks:
                task = Task(
                    task_name=task_dict.get('task_name', 'Unnamed Task'),
                    deadline=task_dict.get('deadline', 'Not specified'),
                    estimated_hours=float(task_dict.get('estimated_hours', 2)),
                    priority=task_dict.get('priority', 'Medium'),
                    course=task_dict.get('course'),
                    scheduled_time=None,
                    status='pending'
                )
                extracted_tasks.append(task)
            
            state["extracted_tasks"] = extracted_tasks
            state["messages"].append(f"Extracted {len(extracted_tasks)} task(s)")
            
            for i, task in enumerate(extracted_tasks, 1):
                state["messages"].append(
                    f"   {i}. {task['task_name']} - Due: {task['deadline']} "
                    f"({task['estimated_hours']}h, {task['priority']})"
                )
        
        else:
            state["messages"].append("No content to process")
            state["status"] = "error"
        
        return state