# main.py (FastAPI Backend)
import os
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate # Import PromptTemplate
from dotenv import load_dotenv
import uvicorn
import io
import edge_tts # For Text-to-Speech

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

# --- Configuration ---
PDF_PATH = "Hotel_Details.pdf"  # Replace with the path to your PDF file
VECTOR_DB_PATH = "faiss_index" # Directory to save/load FAISS index
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct" # or "mixtral-8x7b-32768", "llama3-70b-8192", "meta-llama/llama-4-maverick-17b-128e-instruct"

# --- Global Variables for RAG components ---
vector_store = None
qa_chain = None

# --- FastAPI App Initialization ---
app = FastAPI()

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 1. Document Loading and Chunking ---
def load_and_chunk_documents(pdf_path):
    """Loads a PDF and splits it into manageable chunks."""
    print(f"Loading document: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}. Please provide a valid path.")
        return None

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
        vector_store_obj = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
    else:
        print(f"Creating new FAISS index and saving to {vector_db_path}...")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_store_obj = FAISS.from_documents(chunks, embeddings)
        vector_store_obj.save_local(vector_db_path)
        print("FAISS index created and saved.")
    return vector_store_obj

# --- Lifespan Events for FastAPI ---
@app.on_event("startup")
async def startup_event():
    """Initializes RAG components when the FastAPI app starts."""
    global vector_store, qa_chain

    print("Initializing RAG Chatbot components...")

    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY environment variable not set.")
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set.")

    # 1. Load and chunk documents
    chunks = load_and_chunk_documents(PDF_PATH)
    if not chunks:
        raise HTTPException(status_code=500, detail="Failed to load and chunk documents.")

    # 2. Create/Load Vector Store
    vector_store = create_vector_store(chunks, VECTOR_DB_PATH, EMBEDDING_MODEL_NAME)
    if not vector_store:
        raise HTTPException(status_code=500, detail="Failed to create or load vector store.")

    # 3. Initialize Groq LLM
    print(f"Initializing Groq LLM with model: {GROQ_LLM_MODEL}")
    llm = ChatGroq(temperature=0, model_name=GROQ_LLM_MODEL)

    # --- RAG Specific System Prompt for Royal Orchid Hotel Intelligence Assistant (OrchidAI) ---
    # This prompt provides context and instructions to the LLM
    template = """You are OrchidAI, the helpful and friendly intelligence assistant for the Royal Orchid Hotel.
    Your primary goal is to provide accurate and concise information about the Royal Orchid Hotel based ONLY on the provided context.
    If the question cannot be answered using the provided context, politely state that you cannot find the information within the available documents.
    Do not make up information or answer questions outside the scope of the Royal Orchid Hotel.
    Always maintain a professional, courteous, and welcoming tone.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:"""

    # Create a PromptTemplate instance
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


    # 4. Create RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 relevant chunks
        return_source_documents=False, # Not needed for UI, keeps response smaller
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Pass the custom prompt here
    )
    print("RAG Chatbot components initialized.")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML page."""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/chat")
async def chat(request: Request):
    """Handles chat requests, processes with RAG, and returns the answer."""
    if not qa_chain:
        raise HTTPException(status_code=500, detail="Chatbot not initialized.")

    data = await request.json()
    user_question = data.get("message")

    if not user_question:
        raise HTTPException(status_code=400, detail="Message not provided.")

    print(f"Received question: {user_question}")
    try:
        result = qa_chain.invoke({"query": user_question})
        answer = result["result"]
        print(f"Generated answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing your request: {e}")

@app.get("/tts")
async def text_to_speech(text: str):
    """
    Generates speech from text using edge_tts and streams it as an audio file.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text for TTS not provided.")

    print(f"Generating speech for: {text[:50]}...")
    try:
        # Using a male voice (e.g., 'en-US-GuyNeural'), you can explore others.
        # Check available voices: import asyncio; asyncio.run(edge_tts.list_voices())
        # For a more welcoming tone, you might consider a female voice like 'en-US-AriaNeural'
        VOICE = "en-US-SaraNeural" # Changed to a potentially more welcoming female voice
        communicate = edge_tts.Communicate(text, VOICE)
        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)
        print("Speech generated successfully.")
        return StreamingResponse(audio_buffer, media_type="audio/mpeg")
    except Exception as e:
        print(f"Error during TTS generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {e}")

# To run the FastAPI app:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000