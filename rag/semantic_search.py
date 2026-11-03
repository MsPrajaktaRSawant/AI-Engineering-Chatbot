
from pathlib import Path

from pdf_loader import extract_text_from_pdf
from text_chunker import chunk_text
from embedding import generate_embeddings
from faiss_store import create_faiss_index


def search_pdf(query, chunks, index, top_k=3):

    # Generate embedding for the user's question
    query_embedding = generate_embeddings([query])

    # Search FAISS for the most similar vectors
    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    print("\nSEARCH RESULTS")
    print("=" * 40)

    # Display matching text chunks
    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):

        if idx == -1:
            continue

        print(f"\nResult {rank}")
        print(f"Similarity score: {score:.4f}")
        print("Relevant text:")
        print(chunks[idx])


if __name__ == "__main__":

    # Locate project directory
    project_root = Path(__file__).resolve().parent.parent

    # Locate PDF
    pdf_path = (
        project_root
        / "documents"
        / "ai_engineering_rag_practice_guide.pdf"
    )

    # Extract PDF text
    print("Loading PDF...")

    text = extract_text_from_pdf(pdf_path)

    # Create chunks
    print("\nCreating chunks...")

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No text chunks found in the PDF.")

    # Generate document embeddings
    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(chunks)

    # Create FAISS index
    print("\nCreating FAISS index...")

    index = create_faiss_index(embeddings)

    print("FAISS index created!")
    print("Total vectors:", index.ntotal)

    # Ask user for a question
    print("\nYou can now search your PDF!")

    query = input("\nEnter your question: ")

    # Search PDF
    search_pdf(query, chunks, index)
