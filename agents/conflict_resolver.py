import os
import sys
import config
from dotenv import load_dotenv
from utils.tracing import setup_tracing
from langsmith import Client
from typing import Dict, List, Tuple
from agents.state import SchedulerState
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



class ConflictResolverAgent:
    
    def __init__(self):
        self.llm = get_llm(temperature=0.2)
    
    def check_conflicts(self, state: SchedulerState) -> Dict:
        state["messages"].append("Conflict Checker: Analyzing schedule for conflicts...")
        state["current_step"] = "check_conflicts"
        
        schedule = state.get("schedule", [])
        
        if not schedule:
            state["messages"].append("   No schedule to check")
            state["needs_conflict_resolution"] = False
            return state
        
        conflicts = []
        
        overlaps = self._detect_time_overlaps(schedule)
        if overlaps:
            conflicts.extend(overlaps)
        
        deadline_conflicts = self._detect_deadline_violations(schedule, state.get("extracted_tasks", []))
        if deadline_conflicts:
            conflicts.extend(deadline_conflicts)
        
        duration_issues = self._detect_duration_issues(schedule)
        if duration_issues:
            conflicts.extend(duration_issues)
        
        state["conflicts"] = conflicts
        
        if conflicts:
            state["needs_conflict_resolution"] = True
            state["messages"].append(f"Found {len(conflicts)} conflict(s):")
            for conflict in conflicts:
                state["messages"].append(f"      - {conflict}")
        else:
            state["needs_conflict_resolution"] = False
            state["messages"].append("No conflicts detected")
        
        return state
    
    def resolve_conflicts(self, state: SchedulerState) -> Dict:
        state["messages"].append("Conflict Resolver: Fixing conflicts...")
        state["current_step"] = "resolve_conflicts"
        
        conflicts = state.get("conflicts", [])
        schedule = state.get("schedule", [])
        
        if not conflicts or not schedule:
            state["messages"].append("   Nothing to resolve")
            return state
        
        resolution = self._resolve_with_llm(schedule, conflicts, state)
        
        if resolution:
            state["schedule"] = resolution
            state["conflicts"] = []
            state["needs_conflict_resolution"] = False
            state["messages"].append(f"Resolved conflicts, updated schedule")
        else:
            state["messages"].append("Could not fully resolve conflicts")
        
        return state
    
    def _detect_time_overlaps(self, schedule: List[Dict]) -> List[str]:
        conflicts = []
        
        sorted_schedule = sorted(schedule, key=lambda x: (x['date'], x['time_slot']))
        
        for i in range(len(sorted_schedule) - 1):
            current = sorted_schedule[i]
            next_item = sorted_schedule[i + 1]
            
            if current['date'] == next_item['date']:
                try:
                    current_end = current['time_slot'].split(' - ')[1]
                    next_start = next_item['time_slot'].split(' - ')[0]
                    
                    if current_end > next_start:
                        conflicts.append(
                            f"Overlap on {current['date']}: "
                            f"{current['task_name']} ({current['time_slot']}) and "
                            f"{next_item['task_name']} ({next_item['time_slot']})"
                        )
                except:
                    pass
        
        return conflicts
    
    def _detect_deadline_violations(self, schedule: List[Dict], tasks: List[Dict]) -> List[str]:
        """Detect tasks scheduled after their deadline."""
        conflicts = []
        
        task_deadlines = {task['task_name']: task['deadline'] for task in tasks}
        
        for slot in schedule:
            task_name = slot['task_name']
            scheduled_date = slot['date']
            
            if task_name in task_deadlines:
                deadline = task_deadlines[task_name]
                
                try:
                    sched_dt = datetime.strptime(scheduled_date, '%Y-%m-%d')
                    dead_dt = datetime.strptime(deadline, '%Y-%m-%d')
                    
                    if sched_dt > dead_dt:
                        conflicts.append(
                            f"Deadline violation: {task_name} scheduled on {scheduled_date} "
                            f"but due on {deadline}"
                        )
                except:
                    pass
        
        return conflicts
    
    def _detect_duration_issues(self, schedule: List[Dict]) -> List[str]:
        """Detect unrealistic durations (too long)."""
        conflicts = []
        
        for slot in schedule:
            duration = slot.get('duration_hours', 0)
            
            if duration > 5:
                conflicts.append(
                    f"Duration issue: {slot['task_name']} scheduled for {duration}h "
                    f"(consider breaking into smaller sessions)"
                )
        
        return conflicts
    
    def _resolve_with_llm(self, schedule: List[Dict], conflicts: List[str], state: SchedulerState) -> List[Dict]:
        """Use LLM to resolve conflicts intelligently."""
        
        prompt = f"""You are a scheduling conflict resolver. Fix the following conflicts in the schedule.

CURRENT SCHEDULE:
{json.dumps(schedule, indent=2)}

CONFLICTS DETECTED:
{chr(10).join(f'- {c}' for c in conflicts)}

RESOLUTION RULES:
1. For time overlaps: Shift one task to a different time slot
2. For deadline violations: Move task earlier to meet deadline
3. For duration issues: Split long tasks into multiple shorter sessions
4. Maintain task priorities (High priority gets better slots)
5. Keep 15-min breaks between tasks

OUTPUT FORMAT (return ONLY valid JSON array):
[
  {{
    "task_name": "Task name",
    "date": "YYYY-MM-DD",
    "time_slot": "HH:MM - HH:MM",
    "duration_hours": 2.0,
    "priority": "High",
    "notes": "Reason for change"
  }}
]

Return ONLY the resolved schedule as JSON array, no other text.
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            import re
            content = response.content
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if json_match:
                resolved = json.loads(json_match.group())
                return resolved
            else:
                return None
        
        except Exception as e:
            state["messages"].append(f"Error resolving conflicts: {e}")
            return None


