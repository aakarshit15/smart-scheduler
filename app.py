import streamlit as st
import os
import sys
import config
from agents.graph import SchedulerGraph
from agents.state import create_initial_state
from datetime import datetime
import json

st.set_page_config(
    page_title="Smart Scheduler - AI-Powered Timetable",
    page_icon="",
    layout="wide"
)

sys.path.append(os.path.dirname(__file__))
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .task-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .schedule-slot {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
    }
    .priority-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .priority-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .priority-low {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

if 'schedule_generated' not in st.session_state:
    st.session_state.schedule_generated = False
if 'final_state' not in st.session_state:
    st.session_state.final_state = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'original_input' not in st.session_state:
    st.session_state.original_input = ""
if 'original_files' not in st.session_state:
    st.session_state.original_files = None


st.markdown('<div class="main-header">Smart Scheduler</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Input Your Tasks")
    
    raw_input = st.text_area(
        "Describe your tasks:",
        placeholder="E.g., I have a Data Mining project due Jan 28, Theory assignment due Jan 30...",
        height=150
    )
    
    st.subheader("Or Upload Documents/Images")
    uploaded_files = st.file_uploader(
        "Upload multiple (PDFs/TXT) or images",
        type=['pdf', 'txt', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    file_paths = []
    if uploaded_files:
        os.makedirs("data/temp", exist_ok=True)
        for uploaded_file in uploaded_files:
            filepath = f"data/temp/{uploaded_file.name}"
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded {uploaded_file.name}")
            file_paths.append(filepath)
    else:
        file_paths = []
    
    st.markdown("---")
    generate_button = st.button("Generate Schedule", type="primary", use_container_width=True)
    

if generate_button:
    if not raw_input and not uploaded_files:
        st.error("Please provide either text input or upload a file!")
    else:
        st.session_state.schedule_generated = False
        
        with st.spinner("AI Agents are working..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing multi-agent system...")
            progress_bar.progress(10)
            graph = SchedulerGraph()
            compiled = graph.compile()
            
            status_text.text("Preparing input...")
            progress_bar.progress(20)
            # Store originals for chat updates
            st.session_state.original_input = raw_input
            st.session_state.original_files = file_paths

            initial_state = create_initial_state(
                raw_input=raw_input,
                uploaded_file_paths=file_paths
            )
            
            status_text.text("Extracting tasks...")
            progress_bar.progress(40)
            
            status_text.text("Enriching with RAG...")
            progress_bar.progress(60)
            
            status_text.text("Creating schedule...")
            progress_bar.progress(80)
            
            final_state = compiled.invoke(initial_state)
            
            status_text.text("Complete!")
            progress_bar.progress(100)
            
            st.session_state.final_state = final_state
            st.session_state.schedule_generated = True

            st.session_state.schedule = final_state.get("schedule", [])
            
            st.success("Schedule generated successfully!")

if st.session_state.schedule_generated and st.session_state.final_state:
    state = st.session_state.final_state
    
    st.success("Schedule generated successfully!")
    
    st.header("Your Optimized Schedule")
    

    schedule = st.session_state.get("schedule", [])
    
    if schedule:
        from collections import defaultdict
        schedule_by_date = defaultdict(list)
        
        for slot in schedule:
            schedule_by_date[slot['date']].append(slot)
        
        for date in sorted(schedule_by_date.keys()):
            st.subheader(f"{date}")
            
            for slot in schedule_by_date[date]:
                priority_class = f"priority-{slot['priority'].lower()}"
                
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    st.markdown(f"**{slot['time_slot']}**")
                
                with col2:
                    st.markdown(f"**{slot['task_name']}**")
                    if slot.get('notes'):
                        st.caption(slot['notes'])
                
                with col3:
                    st.markdown(f"<span class='{priority_class}'>{slot['priority']}</span>", 
                            unsafe_allow_html=True)
                    st.caption(f"{slot['duration_hours']}h")

        # Add this import at the top of app.py with other imports
        from streamlit_mermaid import st_mermaid

        
    else:
        st.warning("No schedule generated")
    
    st.markdown("---")
    st.subheader("Visual Flow")

    if 'visual_flow_generated' not in st.session_state:
        st.session_state.visual_flow_generated = False
    if 'mermaid_code' not in st.session_state:
        st.session_state.mermaid_code = None

    if st.button("Generate Visual Flow", type="secondary", use_container_width=True):
        with st.spinner("Generating visual flowchart with Gemini Flash..."):
            from visual_flow_generator import generate_mermaid_flowchart
            
            try:
                schedule_data = st.session_state.get("schedule", [])
                
                if schedule_data:
                    mermaid_code = generate_mermaid_flowchart(schedule_data)
                    
                    st.session_state.mermaid_code = mermaid_code
                    st.session_state.visual_flow_generated = True
                    st.success("Visual flow generated successfully!")
                    st.rerun() 
                else:
                    st.error("No schedule found to generate visual flow!")
                
            except Exception as e:
                st.error(f"Error generating visual flow: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if st.session_state.visual_flow_generated and st.session_state.mermaid_code:
        st.subheader("Schedule Flow Diagram")
        
        try:
            mermaid_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                <script>
                    mermaid.initialize({{
                        startOnLoad: true,
                        theme: 'default',
                        flowchart: {{
                            useMaxWidth: false,
                            htmlLabels: true,
                            curve: 'basis',
                            padding: 20,
                            nodeSpacing: 50,
                            rankSpacing: 80
                        }}
                    }});
                </script>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        padding: 10px;
                        background-color: #f8f9fa;
                        margin: 0;
                        overflow-x: auto;
                    }}
                    .mermaid {{
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        display: inline-block;
                        min-width: 100%;
                    }}
                    .mermaid svg {{
                        max-width: none !important;
                        height: auto !important;
                    }}
                </style>
            </head>
            <body>
                <div class="mermaid">
    {st.session_state.mermaid_code}
                </div>
            </body>
            </html>
            """
            
            st.components.v1.html(mermaid_html, height=900, scrolling=True)
            
            st.info("The diagram flows left-to-right. Scroll horizontally within the diagram to see all dates.")
            
        except Exception as e:
            st.error(f"Rendering error: {e}")
            st.warning("Showing raw Mermaid code instead:")
            st.code(st.session_state.mermaid_code, language="mermaid")
            st.info("You can copy the code above and paste it into https://mermaid.live to view the diagram")
        
        with st.expander("View Mermaid Code"):
            st.code(st.session_state.mermaid_code, language="mermaid")
