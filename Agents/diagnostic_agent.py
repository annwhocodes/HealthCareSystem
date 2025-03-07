from langchain.agents import initialize_agent, Tool
from diagnostic_tools import faiss_tool, medical_search_tool  # Import your tools
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from google.generativeai import GenerativeModel

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

# Define the tools
tools = [faiss_tool, medical_search_tool]

# Initialize the Diagnostics Agent
diagnostics_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",  # Use a simple agent type
    verbose=True  # Print detailed logs
)

# Function to handle user queries
def handle_query(query):
    response = diagnostics_agent.run(query)
    return response

# Example usage
if __name__ == "__main__":
    query = input("Enter your medical query: ")
    result = handle_query(query)
    print(f"Diagnostics Agent Response:\n{result}")