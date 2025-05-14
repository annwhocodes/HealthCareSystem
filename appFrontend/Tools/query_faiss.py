from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize Hugging Face Embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
huggingface_embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Load FAISS Index with Hugging Face Embeddings
vector_store = FAISS.load_local(
    "C:\\Users\\Ananya\\Desktop\\Hackathon_Project\\faiss_index",  # Path to the FAISS index
    embeddings=huggingface_embeddings,
    allow_dangerous_deserialization=True  # Allow loading of pickle files
)

# Function to Query FAISS
def query_vector_store(query):
    # Generate query embedding
    query_embedding = huggingface_embeddings.embed_query(query)
    
    # Perform similarity search
    results = vector_store.similarity_search_by_vector(query_embedding, k=3)
    
    # Display results
    print("Top 3 Matches:")
    for idx, doc in enumerate(results):
        print(f"{idx+1}. {doc.page_content}\n")
