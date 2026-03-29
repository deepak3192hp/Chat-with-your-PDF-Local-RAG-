import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# 1. Load your PDF (Ensure the file is in the same folder as this script)
loader = PyPDFLoader(r"C:\Users\hp\Documents\Training\TATA_PUNCH\INDIANTATA_2026-01-10104352AM.pdf")
data = loader.load()

# 2. Split into chunks so the "small" model can process it
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(data)

# 3. Setup Embeddings (The search engine)
# Make sure you ran: ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Create Vector Store (This saves the data in a local folder 'db')
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 5. Setup LLM (The brain)
# Make sure you ran: ollama pull llama3.2:1b
llm = Ollama(model="llama3.2:1b")

# 6. Create the RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm, 
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# 7. Ask a Question
query = "What the document is telling me ?"
response = qa_chain.invoke(query)

print("\n--- ANSWER ---")
print(response['result'])
