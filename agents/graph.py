import os
import sys
from dotenv import load_dotenv
from langsmith import Client
import config
from agents.state import SchedulerState, create_initial_state
from agents.task_extractor import TaskExtractorAgent
from agents.scheduler_agent import SchedulerAgent
from agents.conflict_resolver import ConflictResolverAgent
from langgraph.graph import StateGraph, END
from typing import Dict


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "smart-scheduler"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_LANGGRAPH_TRACING"] = "true"

try:
    client = Client()
    print("LangSmith tracing initialized")
except Exception as e:
    print(f"LangSmith warning: {e}")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class SchedulerGraph:
    
    def __init__(self):
        self.graph = StateGraph(SchedulerState)
        
        self.task_extractor = TaskExtractorAgent()
        self.scheduler = SchedulerAgent()
        self.conflict_resolver = ConflictResolverAgent()
        
        self._build_graph()
    
    def _build_graph(self):
        
        self.graph.add_node("extract_tasks", self.extract_tasks_node)
        self.graph.add_node("enrich_with_rag", self.enrich_with_rag_node)
        self.graph.add_node("schedule", self.schedule_node)
        self.graph.add_node("check_conflicts", self.check_conflicts_node)
        self.graph.add_node("resolve_conflicts", self.resolve_conflicts_node)
        self.graph.add_node("finalize", self.finalize_node)
        
        self.graph.set_entry_point("extract_tasks")
        self.graph.add_edge("extract_tasks", "enrich_with_rag")
        self.graph.add_edge("enrich_with_rag", "schedule")
        self.graph.add_edge("schedule", "check_conflicts")
        
        self.graph.add_conditional_edges(
            "check_conflicts",
            self.should_resolve_conflicts,
            {
                "resolve": "resolve_conflicts",
                "finalize": "finalize"
            }
        )
        
        self.graph.add_edge("resolve_conflicts", "finalize")
        self.graph.add_edge("finalize", END)
        
        print("Graph built with all 3 agents")
    
    def extract_tasks_node(self, state: SchedulerState) -> Dict:
        return self.task_extractor.process(state)
    
    def enrich_with_rag_node(self, state: SchedulerState) -> Dict:
        return self.scheduler.enrich_with_rag(state)
    
    def schedule_node(self, state: SchedulerState) -> Dict:
        return self.scheduler.create_schedule(state)
    
    def check_conflicts_node(self, state: SchedulerState) -> Dict:
        return self.conflict_resolver.check_conflicts(state)
    
    def resolve_conflicts_node(self, state: SchedulerState) -> Dict:
        return self.conflict_resolver.resolve_conflicts(state)
    
    def finalize_node(self, state: SchedulerState) -> Dict:
        state["messages"].append("Finalizing schedule...")
        state["current_step"] = "finalize"
        state["status"] = "success"
        
        schedule = state.get("schedule", [])
        if schedule:
            formatted = "**Your Schedule:**\n\n"
            for slot in schedule:
                formatted += f"• {slot['date']} {slot['time_slot']}: {slot['task_name']} ({slot['duration_hours']}h)\n"
            state["final_schedule"] = formatted
        
        return state
    
    def should_resolve_conflicts(self, state: SchedulerState) -> str:
        if state.get("needs_conflict_resolution", False):
            return "resolve"
        return "finalize"
    
    def compile(self):
        app = self.graph.compile()
        app.name = "Smart-Scheduler-Multi-Agent"
        return app