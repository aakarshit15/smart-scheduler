import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import config
from collections import defaultdict

def get_gemini_flash():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.7
    )

def generate_mermaid_flowchart(schedule_data):
    llm = get_gemini_flash()
    
    schedule_by_date = defaultdict(list)
    for slot in schedule_data:
        schedule_by_date[slot['date']].append(slot)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at creating clean, organized Mermaid flowcharts with efficient space usage."),
        ("user", """Create a Mermaid flowchart showing this schedule as a HORIZONTAL LEFT-TO-RIGHT timeline grouped by dates.

Schedule: {schedule}

Requirements:
1. Use 'graph LR' (left to right, NOT top to bottom)
2. Group tasks by date using subgraphs
3. Each date should be a column/section
4. Within each date, stack tasks vertically
5. Use different node styles for priority:
   - High priority: rounded boxes like (Task Name)
   - Medium priority: regular boxes like [Task Name]
   - Low priority: stadium boxes like ([Task Name])
6. Keep labels SHORT - use time + first few words of task name
7. Connect dates with arrows between subgraphs
8. Make it compact and space-efficient

Return ONLY the Mermaid syntax starting with 'graph LR', no explanations or markdown.""")
    ])
    
    chain = prompt | llm
    
    schedule_text = ""
    for date in sorted(schedule_by_date.keys()):
        schedule_text += f"\n{date}:\n"
        for slot in schedule_by_date[date]:
            schedule_text += f"  - {slot['time_slot']}: {slot['task_name']} (Priority: {slot['priority']}, {slot['duration_hours']}h)\n"
    
    response = chain.invoke({"schedule": schedule_text})
    
    content = response.content
    
    if isinstance(content, list):
        content = content[0].get('text', '') if content else ''
    
    content = str(content).strip()
    if content.startswith("```mermaid"):
        content = content.replace("```mermaid", "").replace("```", "").strip()
    elif content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join([line for line in lines if not line.startswith("```")])
    
    content = content.replace('\\n', '\n')
    
    return content.strip()
