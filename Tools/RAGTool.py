import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings  # Use Hugging Face embeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiLLM(LLM):
    def __init__(self, model_name="gemini-1.5-flash-latest"):
        self.model_name = model_name

    def _call(self, prompt, stop=None):
        """Invoke the Gemini model to generate a response."""
        if isinstance(prompt, (list, tuple)):
            prompt = " ".join([str(p) for p in prompt])
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        response = genai.generate_content(model=self.model_name, contents=prompt)
        return response[0]['text'] if response else ""

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    @property
    def _llm_type(self):
        return "gemini"

class RAGTool:
    def __init__(self, csv_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2", gemini_api_key=None):
        self.csv_path = csv_path
        self.embedding_model = embedding_model
        self.gemini_api_key = gemini_api_key
        self.qa_chain = self._setup_rag_pipeline()

    def _load_csv(self):
        """Load and process the CSV file."""
        df = pd.read_csv(self.csv_path)
        
        # Ensure required columns exist
        if "Medicine Name" in df.columns and "Uses" in df.columns:
            # Combine relevant columns into a single text column
            df["combined_text"] = df.apply(lambda row: f"Medicine: {row['Medicine Name']}, Uses: {row['Uses']}", axis=1)
        else:
            raise ValueError("Expected columns not found in CSV. Check column names.")

        # Convert to a list of texts
        texts = df["combined_text"].tolist()
        return texts

    def _create_vector_store(self, texts):
        """Generate embeddings and store in a vector database."""
        # Use Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store

    def _setup_rag_pipeline(self):
        """Set up the RAG pipeline with Gemini."""
        # Initialize Gemini LLM
        genai.configure(api_key=self.gemini_api_key)
        gemini_llm = GeminiLLM()

        # Load CSV data and create vector store
        texts = self._load_csv()
        vector_store = self._create_vector_store(texts)

        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=gemini_llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        return qa_chain

    def query(self, question):
        """Query the RAG pipeline."""
        result = self.qa_chain({"query": question})
        return result["result"], result["source_documents"]

# Example usage
csv_path = "C:\\Users\\Ananya\\Desktop\\Hackathon_Project\\Data\\Medicine_Details.csv"  
gemini_api_key = os.getenv("GEMINI_API_KEY")
rag_tool = RAGTool(csv_path, gemini_api_key=gemini_api_key)
response, sources = rag_tool.query("What is the use of Paracetamol?")
print("Response:", response)
print("Sources:", sources)