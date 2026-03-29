This app is a Local RAG (Retrieval-Augmented Generation) application built with Streamlit, LangChain, and Ollama. It allows you to have a private, AI-powered conversation with any PDF document without your data ever leaving your computer.
How it Works (The Technical Flow)
Ingestion: When you upload a PDF, the app uses PyPDFLoader to read the text and RecursiveCharacterTextSplitter to break it into small, manageable "chunks."
Embedding: The nomic-embed-text model converts those text chunks into mathematical vectors (numbers) and stores them in a local database called ChromaDB.
Retrieval: When you type "please explain me the document," the app searches the database for the most relevant sections of your PDF.
Generation: The app sends your question + the relevant PDF sections to the Llama 3.2 1B model. The AI then writes a summary based only on that specific context.


<img width="720" height="472" alt="image" src="https://github.com/user-attachments/assets/6227ea5c-6e24-4797-aa3c-9172747f3103" />
