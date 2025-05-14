import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import google.generativeai as genai
import os
from dotenv import load_dotenv
from utils.pdf_reader import extract_text_from_pdf

load_dotenv()

class GeminiLLM:
    def __init__(self, model_name="gemini-1.5-flash-latest"):
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

class RAGTool:
    def __init__(self, pdf_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2", gemini_api_key=None):
        self.pdf_path = pdf_path
        self.embedding_model = embedding_model
        self.gemini_api_key = gemini_api_key
        self.qa_chain = self._setup_rag_pipeline()

    def _load_pdf(self):
        """Load and process the PDF file."""
        texts = extract_text_from_pdf(self.pdf_path)
        if not texts:
            raise ValueError("No text found in the PDF. Please check the file.")
        return texts

    def _create_vector_store(self, texts):
        """Generate embeddings and store in a vector database."""
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store

    def _setup_rag_pipeline(self):
        """Set up the RAG pipeline with Gemini."""
        gemini_llm = GeminiLLM(api_key=self.gemini_api_key)

        # Load PDF data and create vector store
        texts = self._load_pdf()
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
pdf_path = "C:\\Users\\Ananya\\Desktop\\Hackathon_Project\\Data\\sample.pdf"
gemini_api_key = os.getenv("GEMINI_API_KEY")
rag_tool = RAGTool(pdf_path, gemini_api_key=gemini_api_key)
response, sources = rag_tool.query("What is the diagnosis?")
print("Response:", response)
print("Sources:", sources)