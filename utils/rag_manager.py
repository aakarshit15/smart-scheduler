# utils/rag_manager.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from typing import List, Dict
import json
from datetime import datetime
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class RAGManager:
    
    def __init__(self, persist_directory="./data/chroma_db"):
        self.persist_directory = persist_directory
        
        os.makedirs(persist_directory, exist_ok=True)
        
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embeddings model loaded")
        
        self.vectorstore = Chroma(
            collection_name="scheduler_memory",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
    
    def add_task_completion(self, task: Dict):
        text = f"""
Task: {task.get('task_name', 'Unknown')}
Course: {task.get('course', 'General')}
Estimated Hours: {task.get('estimated_hours', 0)}
Actual Hours: {task.get('actual_hours', task.get('estimated_hours', 0))}
Priority: {task.get('priority', 'Medium')}
Completion Status: {task.get('status', 'Completed')}
Notes: {task.get('notes', 'None')}
"""
        
        doc = Document(
            page_content=text,
            metadata={
                "task_name": task.get('task_name', 'Unknown'),
                "course": task.get('course', 'General'),
                "estimated_hours": task.get('estimated_hours', 0),
                "actual_hours": task.get('actual_hours', 0),
                "priority": task.get('priority', 'Medium'),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        self.vectorstore.add_documents([doc])
        print(f"Added task to memory: {task.get('task_name', 'Unknown')}")
    
    def add_schedule_pattern(self, pattern: Dict):
        text = f"""
Time Slot: {pattern.get('time_slot', 'Unknown')}
Day Type: {pattern.get('day_type', 'Weekday')}
Productivity Level: {pattern.get('productivity', 'Medium')}
Task Type Completed: {pattern.get('task_type', 'General')}
Success Rate: {pattern.get('success_rate', 0)}%
"""
        
        doc = Document(
            page_content=text,
            metadata=pattern
        )
        
        self.vectorstore.add_documents([doc])
        print(f"Added schedule pattern to memory")
    
    def retrieve_similar_tasks(self, task_description: str, k: int = 3) -> List[Dict]:
        results = self.vectorstore.similarity_search_with_score(
            task_description, 
            k=k
        )
        
        similar_tasks = []
        for doc, score in results:
            similar_tasks.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            })
        
        print(f"Found {len(similar_tasks)} similar task(s)")
        return similar_tasks
    
    def get_best_time_slots(self, task_type: str, k: int = 3) -> List[Dict]:
        query = f"Productive time for {task_type} tasks"
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        time_slots = []
        for doc, score in results:
            time_slots.append({
                "metadata": doc.metadata,
                "relevance_score": float(score)
            })
        
        return time_slots
    
    def seed_initial_data(self):
        print("Seeding initial productivity data...")
        
        sample_tasks = [
            {
                "task_name": "Data Mining Assignment",
                "course": "Data Mining",
                "estimated_hours": 3,
                "actual_hours": 4,
                "priority": "High",
                "status": "Completed",
                "notes": "Took longer due to debugging"
            },
            {
                "task_name": "Theory of Computation Problem Set",
                "course": "Theory of Computation",
                "estimated_hours": 2,
                "actual_hours": 2.5,
                "priority": "High",
                "status": "Completed",
                "notes": "Complex proofs required extra time"
            },
            {
                "task_name": "Database Lab Report",
                "course": "Database Systems",
                "estimated_hours": 2,
                "actual_hours": 1.5,
                "priority": "Medium",
                "status": "Completed",
                "notes": "Finished faster than expected"
            },
            {
                "task_name": "Graph Algorithms Implementation",
                "course": "Advanced Algorithms",
                "estimated_hours": 4,
                "actual_hours": 5,
                "priority": "High",
                "status": "Completed",
                "notes": "Implementation was complex"
            }
        ]
        
        for task in sample_tasks:
            self.add_task_completion(task)
        
        sample_patterns = [
            {
                "time_slot": "9 AM - 12 PM",
                "day_type": "Weekday",
                "productivity": "High",
                "task_type": "Coding/Implementation",
                "success_rate": 85
            },
            {
                "time_slot": "2 PM - 5 PM",
                "day_type": "Weekday",
                "productivity": "Medium",
                "task_type": "Reading/Theory",
                "success_rate": 70
            },
            {
                "time_slot": "7 PM - 10 PM",
                "day_type": "Weekday",
                "productivity": "Medium",
                "task_type": "Assignments",
                "success_rate": 75
            },
            {
                "time_slot": "10 AM - 1 PM",
                "day_type": "Weekend",
                "productivity": "High",
                "task_type": "Projects",
                "success_rate": 90
            }
        ]
        
        for pattern in sample_patterns:
            self.add_schedule_pattern(pattern)
        
        print("Initial data seeded successfully!")