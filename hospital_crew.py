from crewai import Agent, Task, Crew
from diagnostic_tools import faiss_tool, medical_search_tool  # Import your tools
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

##create a tool -> it will be a python function (it will accept the path of the pdf as its input and it will return the name of the columns or the info of the dataset. (df.info, df.columns))
def read_df(path):
    """To read a csv or excel file and return the name of the columns"""
    df = pd.read_csv(path)
    columns = df.columns
    return columns
# Custom LLM wrapper for Gemini (Not required)
class GeminiLLM:
    def __init__(self, model_name="gemini-pro"):
        self.model = genai.GenerativeModel(model_name)

    def invoke(self, prompt, config=None, **kwargs):
        """Invoke the Gemini model to generate a response."""
        # Ensure the input is a string
        if isinstance(prompt, (list, tuple)):
            prompt = " ".join([str(p) for p in prompt])
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        # Generate the response
        response = self.model.generate_content(prompt)
        return response.text

    def __call__(self, prompt):
        """Alias for invoke to maintain backward compatibility."""
        return self.invoke(prompt)

# Master Agent
master_agent = Agent(
    role="Master Agent",
    goal="Assign tasks to other agents and retrieve information for the user.",
    backstory="You are the central orchestrator of the MediMind system, ensuring smooth coordination between agents.",
    llm=GeminiLLM(model_name="gemini-pro"),  # Use Gemini as the LLM
    verbose=True,
    allow_delegation=True
)

# Diagnostics Agent
diagnostics_agent = Agent(
    role="Diagnostics Agent",
    goal="Provide preliminary diagnoses based on medical PDFs.",
    backstory="You are an AI trained to extract medical insights from PDFs using Retrieval-Augmented Generation (RAG).",
    tools=[faiss_tool, medical_search_tool],  # Use the FAISS query tool and medical web search tool
    llm=GeminiLLM(model_name="gemini-pro"),  # Use Gemini as the LLM
    verbose=True
)

# Search Agent
search_agent = Agent(
    role="Search Agent",
    goal="Retrieve additional medical information from trusted online sources.",
    backstory="You are an AI specialized in searching the internet for reliable medical data.",
    tools=[medical_search_tool],  # Use the medical web search tool
    llm=GeminiLLM(model_name="gemini-pro"),  # Use Gemini as the LLM
    verbose=True
)

# Hospital Management Agent
hospital_management_agent = Agent(
    role="Hospital Management Agent",
    goal="Manage patient records and hospital data, and visualize insights.",
    backstory="You are an AI specialized in processing and visualizing hospital data.",
    tools=[],  # Add a PythonREPLTool or custom tool for Excel processing
    llm=GeminiLLM(model_name="gemini-pro"),  # Use Gemini as the LLM
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
    description="Manage patient records and hospital data, and generate visualizations. Path od the csv is as follows: {excel_path}. Write a code to answer user's question by writing python code using pandas library by using the column names. User's question is as follows: {user_query}.",
    agent=hospital_management_agent,
    expected_output="Insights and visualizations from hospital operations data."
)

# Define the Crew
crew = Crew(
    agents=[master_agent, diagnostics_agent, search_agent, hospital_management_agent],
    tasks=[diagnose_task, search_task, manage_hospital_data_task],
    verbose=True  # Enable detailed logging
)

# Example usage
if __name__ == "__main__":
    user_query = "I have a fever, cough, and fatigue. What could be causing this?"
    pdf_path = r"C:\Users\Ananya\Desktop\Hackathon_Project\Data\Sample_Patient_Report.pdf"  # Use raw string for file path
    excel_path = r"C:\Users\Ananya\Desktop\Hackathon_Project\Data\Hospital_Data.xlsx"  # Use raw string for file path

    # Run the workflow
    result = crew.kickoff(inputs={"user_query": user_query, "pdf_path": pdf_path, "excel_path": excel_path})
    print(f"Final Result:\n{result}")