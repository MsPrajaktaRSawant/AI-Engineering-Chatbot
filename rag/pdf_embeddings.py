
from pathlib import Path

from .pdf_loader import load_all_pdfs
from .text_chunker import chunk_all_documents
from .embedding import generate_embeddings


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    documents_folder = project_root / "documents"

    print("Loading all PDF documents...")

    documents = load_all_pdfs(documents_folder)

    if not documents:
        raise ValueError(
            "No PDF content found. Check your documents folder."
        )

    print("Total pages loaded:", len(documents))

    print("\nCreating text chunks...")

    all_chunks = chunk_all_documents(documents)

    if not all_chunks:
        raise ValueError("No text chunks were created.")

    print("Total chunks:", len(all_chunks))

    print("\nGenerating embeddings...")

    texts = [chunk["text"] for chunk in all_chunks]

    embeddings = generate_embeddings(texts)

    print("\nALL PDF EMBEDDINGS GENERATED SUCCESSFULLY!")

    print("Total pages:", len(documents))
    print("Total chunks:", len(all_chunks))
    print("Embedding shape:", embeddings.shape)

    print("\nFirst chunk source:")
    print(all_chunks[0]["source"])

    print("\nFirst chunk page:")
    print(all_chunks[0]["page"])

    print("\nFirst chunk text:")
    print(all_chunks[0]["text"])