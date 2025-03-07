import sys
import os
from crewai import Agent, Task, Crew
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd

# Add the Tools directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Tools'))

from Tools.repl_tool import PythonREPLTool  # Import Python REPL tool
from Tools.visualiser_tool import VisualiserTool  # Import Visualiser tool
from Tools.csv_reader_tool import CSVReaderTool  # Import CSV Reader tool

# Load environment variables
load_dotenv()

# Initialize Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Custom LLM wrapper for Gemini
class GeminiLLM:
    def __init__(self, model_name="gemini-1.5-flash-latest"):
        self.model = genai.GenerativeModel(model_name)

    def invoke(self, prompt, config=None, **kwargs):
        """Invoke the Gemini model to generate a response."""
        if isinstance(prompt, (list, tuple)):
            prompt = " ".join([str(p) for p in prompt])
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        response = self.model.generate_content(prompt)
        return response.text

    def __call__(self, prompt):
        """Alias for invoke to maintain backward compatibility."""
        return self.invoke(prompt)

# Define the CSV path
csv_path = r"C:\Users\Ananya\Desktop\Hackathon_Project\Data\hospital_records_2021_2024_with_bills.csv"  # Use raw string for file path

# Master Agent
master_agent = Agent(
    role="Master Agent",
    goal="Assign tasks to other agents and retrieve information for the user.",
    backstory="You are the central orchestrator of the MediMind system, ensuring smooth coordination between agents.",
    llm=GeminiLLM(model_name="gemini-1.5-flash-latest"),  # Use Gemini as the LLM
    verbose=True,
    allow_delegation=True
)

# Diagnostics Agent
diagnostics_agent = Agent(
    role="Diagnostics Agent",
    goal="Provide preliminary diagnoses based on medical PDFs.",
    backstory="You are an AI trained to extract medical insights from PDFs using Retrieval-Augmented Generation (RAG).",
    tools=[CSVReaderTool(csv_path)],  # Use the CSV Reader tool with csv_path
    llm=GeminiLLM(model_name="gemini-1.5-flash-latest"),  # Use Gemini as the LLM
    verbose=True
)

# Search Agent
search_agent = Agent(
    role="Search Agent",
    goal="Retrieve additional medical information from trusted online sources.",
    backstory="You are an AI specialized in searching the internet for reliable medical data.",
    tools=[CSVReaderTool(csv_path)],  # Use the CSV Reader tool with csv_path
    llm=GeminiLLM(model_name="gemini-1.5-flash-latest"),  # Use Gemini as the LLM
    verbose=True
)

# Hospital Management Agent
hospital_management_agent = Agent(
    role="Hospital Management Agent",
    goal="Manage patient records and hospital data, and visualize insights.",
    backstory="You are an AI specialized in processing and visualizing hospital data.",
    tools=[PythonREPLTool(), VisualiserTool()],  # Add Python REPL tool and Visualiser tool for data processing and visualization
    llm=GeminiLLM(model_name="gemini-1.5-flash-latest"),  # Use Gemini as the LLM
    verbose=True
)

# Task 1: Diagnose Symptoms
diagnose_task = Task(
    description="Analyze the user's symptoms and provide a preliminary diagnosis using medical PDFs. Medical PDFs are: {pdf_path}",
    agent=diagnostics_agent,
    expected_output="A preliminary diagnosis based on the provided PDFs",
)

# Task 2: Search for Additional Information
search_task = Task(
    description="Search trusted online sources for additional medical information.",
    agent=search_agent,
    expected_output="Additional medical information from trusted online sources.",
    context=[diagnose_task]  # Depends on the Diagnostics Agent
)

# Task 3: Manage Hospital Data
manage_hospital_data_task = Task(
    description="Manage patient records and hospital data, and generate visualizations. Path of the CSV is as follows: {csv_path}. Write a code to answer the user's question by writing Python code using the pandas library and the column names. User's question is as follows: {user_query}.",
    agent=hospital_management_agent,
    expected_output="Insights and visualizations from hospital operations data.",
    tools=[PythonREPLTool(), VisualiserTool()]  # Use Python REPL tool and Visualiser tool for this task
)

# Define the Crew
crew = Crew(
    agents=[master_agent, diagnostics_agent, search_agent, hospital_management_agent],
    tasks=[diagnose_task, search_task, manage_hospital_data_task],
    verbose=True  # Enable detailed logging
)

# Example usage
if __name__ == "__main__":
    user_query = "What is the average age of patients in the dataset?"
    pdf_path = r"C:\Users\Ananya\Desktop\Hackathon_Project\Data\Sample_Patient_Report.pdf"  # Use raw string for file path

    # Run the workflow
    result = crew.kickoff(inputs={"user_query": user_query, "pdf_path": pdf_path, "csv_path": csv_path})
    print(f"Final Result:\n{result}")