import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load CSV file
csv_path = r"C:\Users\Ananya\Desktop\Hackathon_Project\Data\Medicine_Details.csv"
df = pd.read_csv(csv_path)

# Ensure required columns exist
if "Medicine Name" in df.columns and "Uses" in df.columns:
    df["combined_text"] = df.apply(lambda row: f"Medicine: {row['Medicine Name']}, Uses: {row['Uses']}", axis=1)
else:
    raise ValueError("Expected columns not found in CSV. Check column names.")

# Convert to a list of texts
texts = df["combined_text"].tolist()

# Generate embeddings manually
embeddings = model.encode(texts)  # This returns a numpy array

# Pair texts with their embeddings as tuples
text_embeddings = list(zip(texts, embeddings))

# Initialize FAISS index correctly
faiss_index = FAISS.from_embeddings(
    text_embeddings=text_embeddings,  # Pass list of (text, embedding) tuples
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# Save FAISS index
faiss_index.save_local("faiss_index")
print("FAISS index created and saved successfully.")