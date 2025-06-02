from crewai import Agent, Task, Crew, tools
from Frontend.repl_tool import PythonREPLTool  # Import your custom tool
from RAGTool1 import RAGTool  # Import your custom tool
from Frontend.webscrappingtool import WebScraper 
from Frontend.PDFSearchTool import PDFSearchTool 
from langchain_community.vectorstores import FAISS
import streamlit as st
from hospital_crew import initialize_crew

# Initialize the crew and store it in session state
if "hospital_crew" not in st.session_state:
    st.session_state.hospital_crew = initialize_crew()

# Redirect to the home page
st.switch_page("pages/home.py")
# Define Custom Tools
class RAGFrameworkTool:
    def __init__(self):
        self.name = "RAGtool"
        self.description = "A tool to retrieve information using the RAG framework."

    def _run(self, query: str) -> str:
        """Retrieve information using the RAG framework."""
        # Add your RAG logic here
        return f"Retrieved information for: {query}"

class WebScrapingTool:
    def __init__(self):
        self.name = "webscrappingtool"
        self.description = "A tool to scrape data from reliable medical websites."

    def _run(self, query: str) -> str:
        """Scrape data from websites."""
        # Add your web scraping logic here
        return f"Scraped data for: {query}"

class InternetSearchTool:
    def __init__(self):
        self.name = "internetsearchtool"
        self.description = "A tool to search the internet for medical information."

    def _run(self, query: str) -> str:
        """Search the internet for information."""
        # Add your internet search logic here
        return f"Searched the internet for: {query}"

# Wrap the tools in Tool objects
rag_framework_tool = tools(
    name="rag_framework_tool",
    func=RAGTool()._run,
    description="A tool to retrieve information using the RAG framework."
)

web_scraping_tool = tools(
    name="web_scraping_tool",
    func=WebScraper()._run,
    description="A tool to scrape data from reliable medical websites."
)

internet_search_tool = tools(
    name="internet_search_tool",
    func=InternetSearchTool()._run,
    description="A tool to search the internet for medical information."
)

# Create an instance of PythonREPLTool
python_repl_tool_instance = PythonREPLTool()

# Wrap the PythonREPLTool in a Tool object
python_repl_tool = tools(
    name=python_repl_tool_instance.name,
    func=python_repl_tool_instance.execute,
    description=python_repl_tool_instance.description
)

# Define the Master Agent
master_agent = Agent(
    role="Master Agent",
    goal="Coordinate and assign tasks to other agents based on user requests.",
    backstory="You are the central controller of the hospital crew. Your job is to delegate tasks to the appropriate agents and ensure smooth communication between them.",
    verbose=True,
    allow_delegation=True
)

# Define the Diagnostic Agent
diagnostic_agent = Agent(
    role="Diagnostic Agent",
    goal="Provide accurate diagnosis for patients based on their test results.",
    backstory="You are an AI-powered diagnostic assistant. You analyze patient test results in PDF format, use a RAG framework to retrieve information from reliable medical sources, and provide a diagnosis.",
    tools=[rag_framework_tool],  # Use the custom tool here
    verbose=True
)

# Define the Search Agent
search_agent = Agent(
    role="Search Agent",
    goal="Provide information about medicines, drugs, diseases, and where to buy medicines at the best prices.",
    backstory="You are an AI-powered search assistant. You use web scraping and internet search tools to retrieve reliable medical information and provide the top 3 websites to buy medicines.",
    tools=[web_scraping_tool, internet_search_tool],  # Use the custom tools here
    verbose=True
)

# Define the Hospital Management Agent
hospital_management_agent = Agent(
    role="Hospital Management Agent",
    goal="Manage patient records and hospital data, and visualize them using Python REPL.",
    backstory="You are an AI-powered hospital management assistant. You handle patient records, hospital data, and provide visualizations for better decision-making.",
    tools=[repl_tool],  # Use the custom tool here
    verbose=True
)

# Define Tasks for Each Agent
diagnostic_task = Task(
    description="Analyze the patient's test results in PDF format and provide a diagnosis using the RAG framework.",
    agent=diagnostic_agent,
    expected_output="A detailed diagnosis report for the patient based on their test results."
)

search_task = Task(
    description="Retrieve information about a medicine, drug, or disease, and provide the top 3 websites to buy the medicine at the best price.",
    agent=search_agent,
    expected_output="A summary of the requested information and a list of the top 3 websites to buy the medicine."
)

hospital_management_task = Task(
    description="Manage patient records and hospital data, and visualize the data using Python REPL.",
    agent=hospital_management_agent,
    expected_output="Visualized patient records and hospital data for better decision-making."
)

# Define the Crew
hospital_crew = Crew(
    agents=[master_agent, diagnostic_agent, search_agent, hospital_management_agent],
    tasks=[diagnostic_task, search_task, hospital_management_task],
    verbose=True
)

# Execute the Crew
if __name__ == "__main__":
    # Example input from the user
    user_request = "I need a diagnosis for a patient with the following test results: [insert PDF link]."

    # Assign the task to the Master Agent
    result = hospital_crew.kickoff(inputs={"user_request": user_request})
    print(result)