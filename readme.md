# 📈 FinSight: AI Financial Analyst Agent

**FinSight** is an autonomous RAG (Retrieval Augmented Generation) agent designed to ingest, analyze, and synthesize complex financial reports (10-K Filings, Annual Reports). 

Built to solve the problem of information overload in financial and legal domains, FinSight allows users to "chat" with dense documents, providing citations for every claim.

## 🚀 Key Features
* **Semantic Search:** Uses Vector Embeddings (`all-MiniLM-L6-v2`) to find relevant context, not just keyword matching.
* **Source Citations:** Every answer cites the specific page number from the PDF to ensure trust and zero hallucinations.
* **Hybrid Tech Stack:** Combines LangChain for orchestration, ChromaDB for vector memory, and Meta's Llama-3 (via Groq) for high-speed inference.
* **Interactive UI:** Full-stack chat interface built with Streamlit.

## 🛠️ Tech Stack
* **LLM:** Llama-3-8b (Groq Inference Engine)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers`)
* **Frontend:** Streamlit
* **Ingestion:** PyPDF, RecursiveCharacterTextSplitter

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/DevGoyal07/FinSight-Agent.git](https://github.com/DevGoyal07/FinSight-Agent.git)
   cd FinSight-Agent

   Create a Virtual Environment

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

Bash

pip install -r requirements.txt
Setup Keys

Create a .env file in the root directory.

Add your Groq API Key:

GROQ_API_KEY=your_key_here
🏃‍♂️ Usage
1. Ingest Data: Place your PDF in the root folder named financial_report.pdf and run:

Bash

python vector_store.py
2. Run the App:

Bash

streamlit run app.py
🔮 Future Roadmap
Integrate GraphRAG (Neo4j) to map complex entity relationships.

