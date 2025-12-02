from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "vector_db"

def test_retrieval():
    print("Loading Embedding Model....")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"Loading Database from {DB_PATH}....")
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

    query = "What are the risks related to technology and AI?"

    print(f"\n Query: {query}\n")
    print("Searching for relevant chunks...")

    results =vector_db.similarity_search(query, k=3)
    print(f"Found {len(results)} matches.\n")

    for i, doc in enumerate(results):
        print(f"--- Result {i+1} (Source: Page{doc.metadata.get('page')}) ---")
        print(doc.page_content[:400] + "....")
        print("\n")

if __name__ == "__main__":
    test_retrieval()

