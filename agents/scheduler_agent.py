import os
import sys
import config
from dotenv import load_dotenv
from utils.tracing import setup_tracing
from langsmith import Client
from typing import Dict, List
from agents.state import SchedulerState, Task
from utils.rag_manager import RAGManager
from utils.llm import get_llm
from datetime import datetime, timedelta
import json


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "smart-scheduler-new"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

try:
    client = Client()
except:
    pass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))







class SchedulerAgent:
    
    def __init__(self):
        self.rag = RAGManager()
        self.llm = get_llm(temperature=0.3) 
    
    def enrich_with_rag(self, state: SchedulerState) -> Dict:
        state["messages"].append("RAG Enrichment Agent: Analyzing past patterns...")
        state["current_step"] = "enrich_with_rag"
        
        tasks = state.get("extracted_tasks", [])
        
        if not tasks:
            state["messages"].append("No tasks to enrich")
            return state
        
        # Retrieve similar past tasks for each task
        all_similar_tasks = []
        
        for task in tasks:
            task_desc = f"{task['task_name']} {task.get('course', '')}"
            similar = self.rag.retrieve_similar_tasks(task_desc, k=2)
            
            if similar:
                all_similar_tasks.extend(similar)
                
                # Adjust estimate based on past data
                avg_actual = sum(s['metadata'].get('actual_hours', 0) for s in similar) / len(similar)
                if avg_actual > 0:
                    # Blend original estimate with historical data
                    adjusted = (task['estimated_hours'] + avg_actual) / 2
                    task['estimated_hours'] = round(adjusted, 1)
                    
                    state["messages"].append(
                        f"   📊 Adjusted {task['task_name']}: {adjusted}h (based on similar tasks)"
                    )
        
        state["similar_past_tasks"] = all_similar_tasks
        
        # Get recommended time slots
        recommended_slots = self.rag.get_best_time_slots("study", k=3)
        state["recommended_time_slots"] = recommended_slots
        
        state["messages"].append(f"✅ RAG enrichment complete with {len(all_similar_tasks)} similar tasks")
        
        return state
    
    def create_schedule(self, state: SchedulerState) -> Dict:
        state["messages"].append("Scheduler Agent: Creating optimal timetable...")
        state["current_step"] = "schedule"
        
        tasks = state.get("extracted_tasks", [])
        
        if not tasks:
            state["messages"].append("No tasks to schedule")
            return state
        
        tasks_summary = self._format_tasks_for_llm(tasks)
        rag_context = self._format_rag_context(state)
        
        prompt = f"""You are an intelligent scheduling assistant. Create an optimal weekly schedule for the following tasks.

TASKS TO SCHEDULE:
{tasks_summary}

PRODUCTIVITY INSIGHTS (from past data):
{rag_context}

SCHEDULING GUIDELINES:
1. Consider task priorities (High priority tasks get better time slots)
2. Use recommended high-productivity time slots when available
3. Break large tasks (>3 hours) into multiple sessions
4. Add 15-min breaks between tasks
5. Consider deadlines - schedule urgent tasks earlier
6. Today's date: {datetime.now().strftime('%Y-%m-%d')}

OUTPUT FORMAT (return ONLY valid JSON array):
[
  {{
    "task_name": "Task name",
    "date": "YYYY-MM-DD",
    "time_slot": "HH:MM - HH:MM",
    "duration_hours": 2.0,
    "priority": "High",
    "notes": "Why this time slot"
  }}
]

Return ONLY the JSON array, no other text.
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            import re
            content = response.content
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if json_match:
                schedule = json.loads(json_match.group())
                state["schedule"] = schedule
                
                state["messages"].append(f"Created schedule with {len(schedule)} time slots")
                
                for task in tasks:
                    task['status'] = 'scheduled'
                
                for i, slot in enumerate(schedule, 1):
                    state["messages"].append(
                        f"   {i}. {slot['date']} {slot['time_slot']}: {slot['task_name']}"
                    )
            
            else:
                state["messages"].append("Failed to parse schedule from LLM")
                state["schedule"] = []
        
        except Exception as e:
            state["messages"].append(f"Error creating schedule: {e}")
            state["schedule"] = []
        
        return state
    
    def _format_tasks_for_llm(self, tasks: List[Task]) -> str:
        lines = []
        for i, task in enumerate(tasks, 1):
            lines.append(
                f"{i}. {task['task_name']}\n"
                f"   Course: {task.get('course', 'General')}\n"
                f"   Deadline: {task['deadline']}\n"
                f"   Estimated Duration: {task['estimated_hours']} hours\n"
                f"   Priority: {task['priority']}"
            )
        return "\n\n".join(lines)
    
    def _format_rag_context(self, state: SchedulerState) -> str:
        lines = []
        
        slots = state.get("recommended_time_slots", [])
        if slots:
            lines.append("Best Time Slots (from past productivity):")
            for slot in slots[:3]:
                meta = slot.get('metadata', {})
                lines.append(
                    f"- {meta.get('time_slot', 'N/A')}: "
                    f"{meta.get('productivity', 'Medium')} productivity "
                    f"({meta.get('success_rate', 0)}% success rate)"
                )
        
        similar = state.get("similar_past_tasks", [])
        if similar:
            lines.append("\nSimilar Past Tasks:")
            for task in similar[:3]:
                meta = task.get('metadata', {})
                lines.append(
                    f"- {meta.get('task_name', 'Unknown')}: "
                    f"Estimated {meta.get('estimated_hours', 0)}h, "
                    f"Actually took {meta.get('actual_hours', 0)}h"
                )
        
        return "\n".join(lines) if lines else "No historical data available"