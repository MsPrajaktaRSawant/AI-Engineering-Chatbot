
import os
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

from .pdf_loader import extract_text_from_pdf
from .text_chunker import chunk_text
from .embedding import generate_embeddings
from .faiss_store import create_faiss_index


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(PROJECT_ROOT / ".env")

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)

MODEL_NAME = "openai/gpt-oss-120b"


# ==========================================
# 2. CREATE RAG PIPELINE
# ==========================================

def create_rag_pipeline():

    # Locate PDF
    pdf_path = (
        PROJECT_ROOT
        / "documents"
        / "ai_engineering_rag_practice_guide.pdf"
    )

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Extract PDF text
    print("Loading PDF...")

    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    # Create text chunks
    print("\nCreating text chunks...")

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No text chunks found.")

    print("Total chunks:", len(chunks))

    # Generate embeddings
    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(chunks)

    # Create FAISS index
    print("\nCreating FAISS index...")

    index = create_faiss_index(embeddings)

    print("RAG pipeline initialized successfully!")

    return chunks, index


# ==========================================
# 3. RETRIEVE RELEVANT PDF CONTENT
# ==========================================

def retrieve_context(question, chunks, index, top_k=3):

    # Generate question embedding
    query_embedding = generate_embeddings([question])

    # Search FAISS index
    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    # Retrieve relevant chunks
    relevant_chunks = []

    for idx in indices[0]:

        if idx != -1:
            relevant_chunks.append(chunks[idx])

    # Combine retrieved content
    context = "\n\n".join(relevant_chunks)

    return context


# ==========================================
# 4. GENERATE NON-STREAMING ANSWER
# ==========================================

def ask_question(question, chunks, index, top_k=3):

    # Retrieve relevant context
    context = retrieve_context(
        question,
        chunks,
        index,
        top_k
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI Engineering tutor. "
                "Answer the user's question using the provided PDF context. "
                "If the answer is not available in the context, "
                "clearly say that the information is not available "
                "in the provided document. Do not invent facts."
            )
        },
        {
            "role": "user",
            "content": f"""
Context from PDF:
{context}

User Question:
{question}

Answer based on the PDF context.
"""
        }
    ]

    # Call Groq
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=False
    )

    return response.choices[0].message.content


# ==========================================
# 5. GENERATE STREAMING ANSWER
# ==========================================

def ask_question_stream(question, chunks, index, top_k=3):

    # Retrieve relevant context
    context = retrieve_context(
        question,
        chunks,
        index,
        top_k
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI Engineering tutor. "
                "Answer the user's question using the provided PDF context. "
                "If the answer is not available in the context, "
                "clearly say that the information is not available "
                "in the provided document. Do not invent facts."
            )
        },
        {
            "role": "user",
            "content": f"""
Context from PDF:
{context}

User Question:
{question}

Answer based on the PDF context.
"""
        }
    ]

    # Call Groq with streaming enabled
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True
    )

    # Yield response piece by piece
    for chunk in response:

        content = chunk.choices[0].delta.content

        if content is not None:
            yield content


# ==========================================
# 6. TEST RAG PIPELINE
# ==========================================

if __name__ == "__main__":

    # Initialize pipeline
    chunks, index = create_rag_pipeline()

    print("\nAI Engineering PDF Chatbot is ready!")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nAI Tutor: ", end="", flush=True)

        # Generate streaming answer
        for content in ask_question_stream(
            question,
            chunks,
            index
        ):
            print(content, end="", flush=True)

        print()
