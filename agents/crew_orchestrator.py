from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_core.runnables import Runnable
from agent_tools import initialize_tools

# Load environment variables
load_dotenv()

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Custom LLM wrapper for Gemini
class GeminiLLM(Runnable):
    def __init__(self, model_name="gemini-1.5-flash-latest"):
        self.model = genai.GenerativeModel(model_name)

    def invoke(self, prompt, config=None, **kwargs):
        if isinstance(prompt, (list, tuple)):
            prompt = " ".join([str(p) for p in prompt])
        elif not isinstance(prompt, str):
            prompt = str(prompt)
        response = self.model.generate_content(prompt)
        return response.text

# Initialize the LLM
llm = GeminiLLM()

# Initialize tools
tools = initialize_tools(llm)

# Define agents
master_agent = Agent(
    role="Master Agent",
    goal="Orchestrate and coordinate tasks between specialized medical agents",
    backstory="""You are the central coordinator of a medical AI system. Your role is to understand user queries, 
    break them down into subtasks, and delegate them to the appropriate specialized agents. You ensure all agents 
    work together efficiently to provide comprehensive medical assistance.""",
    llm=llm,
    verbose=True,
    allow_delegation=True
)

search_agent = Agent(
    role="Search Agent",
    goal="Find and validate medical information from reliable sources",
    backstory="""You are a specialized medical search agent with access to trusted medical databases and websites. 
    You ensure all information provided is accurate and comes from reputable sources. You can search the internet 
    and scrape medical websites for up-to-date information.""",
    llm=llm,
    tools=tools["search"],
    verbose=True
)

diagnostic_agent = Agent(
    role="Diagnostic Agent",
    goal="Analyze medical information and provide preliminary insights",
    backstory="""You are an AI diagnostic assistant trained to analyze medical documents, symptoms, and test results. 
    You use advanced tools to process medical PDFs and provide preliminary medical insights while always 
    recommending professional medical consultation.""",
    llm=llm,
    tools=tools["diagnostic"],
    verbose=True
)

hospital_management_agent = Agent(
    role="Hospital Management Agent",
    goal="Manage and analyze hospital data and records",
    backstory="""You are a hospital management specialist that handles patient records, hospital data, and provides 
    insights for better healthcare delivery. You ensure efficient processing of medical records while maintaining 
    patient privacy.""",
    llm=llm,
    tools=tools["management"],
    verbose=True
)

def create_tasks(user_query):
    """Create tasks based on the user query"""
    
    # Task for the search agent to gather information
    search_task = Task(
        description=f"""
        Search and validate medical information for the query: {user_query}
        1. Use medical search tools to find relevant information
        2. Validate information from trusted sources
        3. Compile comprehensive results
        """,
        agent=search_agent,
        expected_output="Validated medical information from reliable sources"
    )

    # Task for the diagnostic agent to analyze
    diagnostic_task = Task(
        description=f"""
        Analyze medical information and provide insights for: {user_query}
        1. Review search results
        2. Analyze relevant medical documents
        3. Provide preliminary medical insights
        """,
        agent=diagnostic_agent,
        expected_output="Medical analysis and preliminary insights",
        context=[search_task]
    )

    # Task for the hospital management agent
    management_task = Task(
        description=f"""
        Process and analyze relevant hospital data for: {user_query}
        1. Access relevant patient records
        2. Analyze hospital data
        3. Provide data-driven insights
        """,
        agent=hospital_management_agent,
        expected_output="Processed medical records and management insights",
        context=[search_task, diagnostic_task]
    )

    return [search_task, diagnostic_task, management_task]

def process_medical_query(user_query):
    """Main function to process medical queries using the crew"""
    
    # Create the crew with tasks
    tasks = create_tasks(user_query)
    crew = Crew(
        agents=[master_agent, search_agent, diagnostic_agent, hospital_management_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    # Execute the crew workflow
    result = crew.kickoff(inputs={"user_query": user_query})
    return result

if __name__ == "__main__":
    # Example usage
    query = "What are the symptoms and treatments for Type 2 Diabetes?"
    result = process_medical_query(query)
    print(f"Final Result:\n{result}") 