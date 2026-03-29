import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- UI Configuration ---
st.set_page_config(page_title="Local PDF Chat", layout="centered")
st.title("📄 Chat with your PDF (Local RAG)")

# --- Session State Initialization ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing..."):
            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and Split
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = splitter.split_documents(docs)
            
            # Create Vector Store
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            st.session_state.vector_store = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings
            )
            st.success("PDF Indexed!")

# --- Main Chat Interface ---
if st.session_state.vector_store:
    user_input = st.chat_input("Ask a question about your PDF...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Setup RAG Chain
                llm = OllamaLLM(model="llama3.2:1b")
                template = """Answer based ONLY on the context: {context}\nQuestion: {question}"""
                prompt = ChatPromptTemplate.from_template(template)
                
                chain = (
                    {"context": st.session_state.vector_store.as_retriever(), "question": RunnablePassthrough()}

                    | prompt | llm | StrOutputParser()
                )
                
                response = chain.invoke(user_input)
                st.write(response)
else:
    st.info("Please upload and process a PDF to start chatting.")
