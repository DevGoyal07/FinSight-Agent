import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "financial_report.pdf"

def load_and_chunk_data():
    print(f"Starting ingestion for: {PDF_PATH}")

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"The file {PDF_PATH} does not exist.")
    loader = PyPDFLoader(PDF_PATH)
    raw_pages = loader.load()

    print(f"Loaded {len(raw_pages)} raw pages from PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    print("Splitting documents into semantic chunks....")
    chunks = splitter.split_documents(raw_pages)

    print(f"Generated {len(chunks)} chunks.")

    return chunks
if __name__ == "__main__":

    my_chunks = load_and_chunk_data()

    print("\n--- Inspection: CHUNK 10---")
    print(my_chunks[10].page_content)

    print("\n---METADATA (Source Tracking) ---")
    print(my_chunks[10].metadata)
