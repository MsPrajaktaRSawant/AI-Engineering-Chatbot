from pathlib import Path

from pdf_loader import extract_text_from_pdf
from text_chunker import chunk_text
from embedding import generate_embeddings


if __name__ == "__main__":

    # Locate project directory
    project_root = Path(__file__).resolve().parent.parent

    # Locate PDF
    pdf_path = (
        project_root
        / "documents"
        / "ai_engineering_rag_practice_guide.pdf"
    )

    # Step 1: Extract PDF text
    print("Extracting PDF text...")

    text = extract_text_from_pdf(pdf_path)

    # Step 2: Divide text into chunks
    print("\nCreating text chunks...")

    chunks = chunk_text(text)

    print("Total chunks:", len(chunks))

    # Step 3: Generate embeddings
    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(chunks)

    # Step 4: Display results
    print("\nPDF EMBEDDINGS GENERATED")
    print("=" * 40)

    print("Total chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)

    print("\nFirst chunk:")
    print(chunks[0])

    print("\nFirst chunk embedding:")
    print(embeddings[0])