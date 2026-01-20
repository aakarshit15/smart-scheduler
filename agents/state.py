# agents/state.py
from typing import TypedDict, List, Dict, Optional, Annotated
from datetime import datetime
import operator



class Task(TypedDict):
    task_name: str
    deadline: str
    estimated_hours: float
    priority: str  # High, Medium, Low
    course: Optional[str]
    scheduled_time: Optional[str]  # Time slot assigned
    status: Optional[str]  # pending, scheduled, conflict


class SchedulerState(TypedDict):
    raw_input: str
    uploaded_file_paths: Optional[List[str]]
    
    extracted_tasks: List[Task]
    
    similar_past_tasks: List[Dict]
    recommended_time_slots: List[Dict]
    
    schedule: List[Dict] 
    conflicts: List[str]  
    
    messages: Annotated[List[str], operator.add] 
    
    current_step: str
    needs_conflict_resolution: bool
    
    final_schedule: Optional[str]  
    status: str


def create_initial_state(
    raw_input: str = "",
    uploaded_file_paths: Optional[List[str]] = None
) -> SchedulerState:
    return SchedulerState(
        raw_input=raw_input,
        uploaded_file_paths=uploaded_file_paths,
        extracted_tasks=[],
        similar_past_tasks=[],
        recommended_time_slots=[],
        schedule=[],
        conflicts=[],
        messages=[f"State initialized at {datetime.now().strftime('%H:%M:%S')}"],
        current_step="start",
        needs_conflict_resolution=False,
        final_schedule=None,
        status="initialized"
    )