from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_and_chunk_data

DB_PATH = "vector_db"

def create_vector_db():
    print("Fetching chunks from ingest.py....")
    chunks = load_and_chunk_data()

    print("Initializing Embedding Model (This might take a minute)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating and Persisting Vector Database...")
    vector_db = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = DB_PATH
    )

    print(f"Success! Database saved to folder: {DB_PATH}")
    print(f"Stored {len(chunks)} chunks.")
    return vector_db

if __name__ == "__main__":
    create_vector_db()