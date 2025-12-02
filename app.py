import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1. Setup Page Configuration (The Tab title and icon)
st.set_page_config(page_title="FinSight AI", page_icon="📈")

# 2. Load API Keys
load_dotenv()

# Configuration
DB_PATH = "vector_db"

# --- CACHED RESOURCE FUNCTION ---
# Engineering Note: We use @st.cache_resource so we don't reload the 
# heavy embedding model every time you type a new question. 
# This makes the app 100x faster.
@st.cache_resource
def get_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Load existing DB
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    return vector_db

# --- THE UI STRUCTURE ---
st.title("📈 FinSight: The AI Financial Analyst")
st.caption("Powered by Groq Llama-3 & LangChain RAG")

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. The Input Box
if prompt := st.chat_input("Ask a question about the financial report..."):
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- THE BRAIN (RAG PIPELINE) ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤖 *Analyzing data...*")
        
        try:
            # Load Database
            vector_db = get_vector_store()
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            
            # Initialize LLM (Groq Llama-3)
            llm = ChatGroq(model="llama-3.1-8b-instant")
            
            # System Prompt
            system_prompt = (
                "You are a Senior Financial Analyst. "
                "Use the provided context to answer the user's question. "
                "If the answer is not in the context, say 'I cannot find that information in the report.' "
                "Always cite page numbers from the metadata."
                "\n\n"
                "Context: {context}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Create Chain
            # Engineering Note: We define a 'document_prompt' to explicitly force 
            # the page number into the text context so the LLM can see it.
            document_prompt = PromptTemplate.from_template(
                "Content: {page_content}\nSource: Page {page}"
            )
            
            question_answer_chain = create_stuff_documents_chain(
                llm, 
                prompt_template, 
                document_prompt=document_prompt
            )
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)    
            # Run Chain
            response = rag_chain.invoke({"input": prompt})
            answer = response['answer']
            
            # Format Sources
            sources = "\n\n**Sources:**\n"
            for doc in response["context"]:
                sources += f"- Page {doc.metadata.get('page', 'Unknown')}\n"
            
            final_response = answer + sources
            
            # Display Answer
            message_placeholder.markdown(final_response)
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            
        except Exception as e:
            st.error(f"Error: {e}")