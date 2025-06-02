from crewai import Agent, Tool
from langchain_core.runnables import Runnable
import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.generativeai import GenerativeModel

# Import tools from the Tools folder
from Frontend.query_faiss import query_faiss  # Replace with actual import
from Frontend.medical_search_tool import medical_search_tool  # Replace with actual import

# Load environment variables
load_dotenv()

# Custom LLM wrapper for Gemini
class GeminiLLM(Runnable):
    def __init__(self, model_name="gemini-1.5-flash-latest"):
        self.model = GenerativeModel(model_name)

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

# Initialize Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the LLM
llm = GeminiLLM(model_name="gemini-1.5-flash-latest")  # Use Gemini as the LLM

# Wrap the imported functions as CrewAI tools
tools = [
    Tool(
        name="query_faiss",
        func=query_faiss,  # Use the imported query_faiss function
        description="Useful for querying a FAISS vector database for medical information."
    ),
    Tool(
        name="medical_search_tool",
        func=medical_search_tool,  # Use the imported medical_search_tool function
        description="Useful for searching medical information from trusted sources."
    )
]

# Initialize the Diagnostics Agent for CrewAI
diagnostics_agent = Agent(
    role="Diagnostics Agent",
    goal="Provide preliminary diagnoses based on medical PDFs.",
    backstory="You are an AI trained to extract medical insights from PDFs using Retrieval-Augmented Generation (RAG).",
    tools=tools,  # Use the tools defined above
    llm=llm,  # Use the Gemini LLM
    verbose=True  # Print detailed logs
)

# Function to handle user queries
def handle_query(query):
    # Use the Diagnostics Agent to process the query
    response = diagnostics_agent.process(query)
    return response

# Example usage
if __name__ == "__main__":
    query = input("Enter your medical query: ")
    result = handle_query(query)
    print(f"Diagnostics Agent Response:\n{result}")