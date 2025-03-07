from langchain.agents import initialize_agent, Tool
from diagnostic_tools import faiss_tool, medical_search_tool
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.runnables import Runnable
from google.generativeai import GenerativeModel
import google.generativeai as genai

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

# Initialize the LLM for the Manager Agent
manager_llm = GeminiLLM(model_name="gemini-1.5-flash-latest")

# Dummy tool for the Manager Agent
def dummy_tool(query):
    return "This is a dummy tool."

dummy_tool = Tool(
    name="Dummy_Tool",
    description="A dummy tool to satisfy the Manager Agent's tool requirement.",
    func=dummy_tool
)

# Initialize the Manager Agent with the dummy tool
manager_agent = initialize_agent(
    tools=[dummy_tool],  # Pass the dummy tool
    llm=manager_llm,
    agent="zero-shot-react-description",  # Use a simple agent type
    verbose=True  # Print detailed logs
)

# Function to handle user input and delegate tasks
def handle_user_input(user_input, pdf_path):
    # Delegate the task to the Diagnostics Agent
    diagnostics_agent = initialize_diagnostics_agent(pdf_path)
    response = diagnostics_agent.run(user_input)
    return response

# Function to initialize the Diagnostics Agent
def initialize_diagnostics_agent(pdf_path):
    """Initialize the Diagnostics Agent with RAG capabilities."""
    # Load and process the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Initialize Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)

    # Set up the RAG pipeline
    llm = GeminiLLM(model_name="gemini-pro")  # Use Gemini as the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    # Define the tools for the Diagnostics Agent
    tools = [faiss_tool, medical_search_tool]

    # Initialize the Diagnostics Agent
    diagnostics_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",  # Use a simple agent type
        verbose=True  # Print detailed logs
    )

    return diagnostics_agent

# Example usage
if __name__ == "__main__":
    user_input = input("Enter your medical query: ")
    pdf_path = input("Enter the path to your medical PDF: ")  # User provides PDF path
    result = handle_user_input(user_input, pdf_path)
    print(f"Diagnostics Agent Response:\n{result}")