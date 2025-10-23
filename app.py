import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

# --- Configuration ---
PDF_PATH = r"Hotel_Details.pdf"  # Replace with the path to your PDF file
# You can create a dummy PDF for testing or use an existing one.
# For example, download a public PDF and rename it.

VECTOR_DB_PATH = "faiss_index" # Directory to save/load FAISS index

# Hugging Face Embedding Model (local and efficient)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Groq LLM Model (choose based on your needs, Llama3 is powerful and fast)
GROQ_LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct" # or "mixtral-8x7b-32768", "llama3-70b-8192"

# --- 1. Document Loading and Chunking ---
def load_and_chunk_documents(pdf_path):
    """Loads a PDF and splits it into manageable chunks."""
    print(f"Loading document: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}. Please provide a valid path.")
        exit()

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Max characters in a chunk
        chunk_overlap=200,    # Overlap between chunks to maintain context
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# --- 2. Create and Persist Embeddings in FAISS ---
def create_vector_store(chunks, vector_db_path, embedding_model_name):
    """
    Creates a FAISS vector store from document chunks and saves it,
    or loads it if it already exists.
    """
    if os.path.exists(vector_db_path) and os.path.isdir(vector_db_path):
        print(f"Loading existing FAISS index from {vector_db_path}...")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
    else:
        print(f"Creating new FAISS index and saving to {vector_db_path}...")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(vector_db_path)
        print("FAISS index created and saved.")
    return vector_store

# --- Main Chatbot Logic ---
def main():
    print("Initializing RAG Chatbot...")

    # Ensure Groq API key is set
    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please set it before running the script (e.g., export GROQ_API_KEY='your_api_key_here').")
        return

    # 1. Load and chunk documents
    chunks = load_and_chunk_documents(PDF_PATH)
    if not chunks:
        return

    # 2. Create/Load Vector Store
    vector_store = create_vector_store(chunks, VECTOR_DB_PATH, EMBEDDING_MODEL_NAME)
    # ... after vector_store = create_vector_store(...)

    

    # 3. Initialize Groq LLM
    print(f"Initializing Groq LLM with model: {GROQ_LLM_MODEL}")
    llm = ChatGroq(temperature=0, model_name=GROQ_LLM_MODEL)

    # 4. Create RetrievalQA Chain
    # This chain handles retrieving relevant documents and then feeding them to the LLM.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' means put all retrieved docs into the prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}), # Retrieve top 3 relevant chunks
        return_source_documents=True # Optionally return the documents that informed the answer
    )

    print("\nChatbot Ready! Ask me questions about your document. Type 'exit' to quit.")

    # 5. Chat Loop
    while True:
        user_question = input("\nYour Question: ")
        if user_question.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break

        print("Searching and generating response...")
        try:
            result = qa_chain.invoke({"query": user_question})
            print("\nAnswer:", result["result"])
            # if result.get("source_documents"):
            #     print("\n--- Source Documents ---")
            #     for i, doc in enumerate(result["source_documents"]):
            #         print(f"Doc {i+1}: {doc.page_content[:200]}...") # Print first 200 chars
            #         if hasattr(doc.metadata, 'page'):
            #             print(f"  (Page: {doc.metadata['page']})")
            #         elif 'page' in doc.metadata:
            #             print(f"  (Page: {doc.metadata['page']})")
            #     print("------------------------")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your GROQ_API_KEY is correctly set and you have network connectivity.")

if __name__ == "__main__":
    main()