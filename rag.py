import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- NEW IMPORTS (More Stable) ---
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load API Key
load_dotenv()

DB_PATH = "vector_db"

def start_rag_chat():
    print("🧠 Loading Database & Models...")
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model="llama-3.1-8b-instant")

    system_prompt = (
        "You are a Senior Financial Analyst assisting a user. "
        "Use the provided context to answer the question. "
        "If the answer is not in the context, say 'I cannot find that information in the report.' "
        "Always cite the page number if available in the metadata."
        "\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n✅ FinSight Agent Ready! (Type 'exit' to quit)\n")
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
            
        print("🤖 Analyzing report...")
        try:
            response = rag_chain.invoke({"input": query})
            print(f"\nAgent: {response['answer']}")
            
            print("\n[Sources:]")
            for doc in response["context"]:
                page_num = doc.metadata.get('page', 'Unknown')
                source_file = doc.metadata.get('source', 'Unknown')
                print(f"- Page {page_num} of {source_file}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_rag_chat()