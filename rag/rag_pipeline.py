
import os
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

from .pdf_loader import load_all_pdfs
from .text_chunker import chunk_all_documents
from .embedding import generate_embeddings
from .faiss_store import create_faiss_index


# PROJECT CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)

MODEL_NAME = "openai/gpt-oss-120b"


# CREATE RAG PIPELINE

def create_rag_pipeline():

    documents_folder = PROJECT_ROOT / "documents"

    print("Loading all PDF documents...")

    documents = load_all_pdfs(documents_folder)

    if not documents:
        raise ValueError("No PDF content found.")

    print("Creating text chunks...")

    all_chunks = chunk_all_documents(documents)

    if not all_chunks:
        raise ValueError("No chunks created.")

    print("Generating embeddings...")

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = generate_embeddings(texts)

    print("Creating FAISS index...")

    index = create_faiss_index(embeddings)

    print("RAG pipeline initialized successfully!")

    print("Total pages:", len(documents))
    print("Total chunks:", len(all_chunks))

    return all_chunks, index


# RETRIEVE RELEVANT CONTEXT

def retrieve_context(question, all_chunks, index, top_k=3):

    query_embedding = generate_embeddings([question])

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(all_chunks))
    )

    relevant_chunks = []

    for idx in indices[0]:

        if idx != -1:

            chunk = all_chunks[idx]

            source_info = (
                f"Source: {chunk['source']}, "
                f"Page: {chunk['page']}"
            )

            relevant_chunks.append(
                f"{source_info}\n{chunk['text']}"
            )

    context = "\n\n".join(relevant_chunks)

    return context


# PREPARE MESSAGES WITH CONVERSATION MEMORY

def prepare_messages(
    question,
    all_chunks,
    index,
    conversation_history=None,
    top_k=3
):

    # Retrieve relevant PDF context
    context = retrieve_context(
        question,
        all_chunks,
        index,
        top_k
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI Engineering tutor. "
                "Explain technical concepts clearly and simply. "
                "Provide practical examples and Python code when useful. "
                "Use the provided PDF context as your knowledge source. "
                "Use previous conversation messages to understand "
                "follow-up questions and references. "
                "Mention the source PDF and page number when useful. "
                "If the answer is not available in the provided context, "
                "clearly say so. Do not invent facts."
            )
        }
    ]

    # Add previous conversation messages
    if conversation_history:

        for message in conversation_history:

            if message["role"] in ["user", "assistant"]:

                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })

    # Add current question with retrieved context
    messages.append({
        "role": "user",
        "content": f"""
Relevant context retrieved from the PDFs:

{context}

Current question:

{question}

Answer using the retrieved PDF context and previous conversation.
"""
    })

    return messages


# NON-STREAMING RESPONSE

def ask_question(
    question,
    all_chunks,
    index,
    conversation_history=None,
    top_k=3
):

    messages = prepare_messages(
        question,
        all_chunks,
        index,
        conversation_history,
        top_k
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=False
    )

    return response.choices[0].message.content


# STREAMING RESPONSE


def ask_question_stream(
    question,
    all_chunks,
    index,
    conversation_history=None,
    top_k=3
):

    messages = prepare_messages(
        question,
        all_chunks,
        index,
        conversation_history,
        top_k
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True
    )

    for chunk in response:

        content = chunk.choices[0].delta.content

        if content is not None:
            yield content



# TERMINAL CHATBOT TEST

if __name__ == "__main__":

    all_chunks, index = create_rag_pipeline()

    conversation_history = []

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

        answer_parts = []

        for content in ask_question_stream(
            question,
            all_chunks,
            index,
            conversation_history
        ):

            print(content, end="", flush=True)

            answer_parts.append(content)

        print()

        answer = "".join(answer_parts)

        # Save conversation for the next question
        conversation_history.append({
            "role": "user",
            "content": question
        })

        conversation_history.append({
            "role": "assistant",
            "content": answer
        })
