import os
import sys
from typing import List, Dict
import config
from utils.llm import get_llm
from pypdf import PdfReader
from PIL import Image
import base64
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class DocumentProcessor:
    
    def __init__(self):
        self.llm = get_llm(temperature=0)
    
    def process_pdf(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            print(f"Extracted {len(text)} characters from PDF")
            return text.strip()
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return ""
    
    def process_image_with_llm(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            from langchain_groq import ChatGroq
            vision_llm = ChatGroq(
                model="llama-3.2-90b-vision-preview",
                temperature=0
            )
            
            from langchain_core.messages import HumanMessage
            
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Extract all text from this image. Include any to-do items, deadlines, dates, course names, assignments, or tasks. Return only the extracted text, preserving the structure."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            )
            
            response = vision_llm.invoke([message])
            extracted_text = response.content
            
            print(f"Extracted text from image using vision model")
            return extracted_text
            
        except Exception as e:
            print(f"Error processing image with vision: {e}")
            print("   Falling back to basic text extraction prompt...")
            return self._fallback_image_processing(image_path)
    
    def _fallback_image_processing(self, image_path: str) -> str:
        return f"[Image uploaded: {os.path.basename(image_path)}]\nPlease manually describe the content."
    
    def extract_tasks_from_text(self, text: str) -> List[Dict]:
        prompt = f"""You are a task extraction expert. Extract all tasks, assignments, and deadlines from the following text.

    For each task, identify:
    - Task name/title
    - Due date/deadline (extract exact date if available, or "Not specified")
    - Estimated duration in hours (your best estimate based on task complexity)
    - Priority (High/Medium/Low based on proximity to deadline and importance)
    - Course/Subject (if mentioned)

    Text to analyze:
    {text}

    Return ONLY a valid JSON array of tasks in this exact format:
    [
    {{
        "task_name": "Complete assignment 1",
        "deadline": "2026-01-25",
        "estimated_hours": 3,
        "priority": "High",
        "course": "Data Mining"
    }}
    ]

    If no tasks found, return: []
    """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            print(f"\n{'='*50}")
            print("LLM RAW RESPONSE:")
            print(content)
            print(f"{'='*50}\n")
            
            import json
            import re
            
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if json_match:
                tasks = json.loads(json_match.group())
                print(f"Extracted {len(tasks)} task(s)")
                return tasks
            else:
                print("No JSON array found in LLM response")
                print(f"Response content: {content[:500]}")  # Print first 500 chars
                return []
                
        except Exception as e:
            print(f"Error extracting tasks: {e}")
            import traceback
            traceback.print_exc()
            return []


    def process_multiple_files(self, filepaths: List[str]) -> str:
        """Process all files and combine extracted text."""
        combined_text = []
        for filepath in filepaths:
            if filepath.lower().endswith('.pdf'):
                text = self.process_pdf(filepath)
            elif filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = self.process_image_with_llm(filepath)
            else: 
                with open(filepath, 'r') as f:
                    text = f.read()
            combined_text.append(text)
        return "\n\n---\n\n".join(combined_text)