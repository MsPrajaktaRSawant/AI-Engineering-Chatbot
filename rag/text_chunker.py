def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":

    from pdf_loader import extract_text_from_pdf
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    pdf_path = (
        project_root
        / "documents"
        / "ai_engineering_rag_practice_guide.pdf"
    )

    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)

    print("\nCHUNKING COMPLETED")
    print("=" * 40)

    print("Total chunks:", len(chunks))

    for index, chunk in enumerate(chunks[:3], start=1):

        print(f"\nCHUNK {index}")
        print(chunk)