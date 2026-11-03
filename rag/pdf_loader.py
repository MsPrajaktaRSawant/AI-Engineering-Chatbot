from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages_text.append({
                "page": page_number,
                "text": page_text
            })

    return pages_text


def load_all_pdfs(folder_path):
    folder = Path(folder_path)

    # Automatically discover every PDF in the documents folder
    pdf_files = sorted(folder.glob("*.pdf"))

    all_documents = []

    for pdf_path in pdf_files:
        print(f"Processing PDF: {pdf_path.name}")

        pages = extract_text_from_pdf(pdf_path)

        for page_data in pages:
            all_documents.append({
                "source": pdf_path.name,
                "page": page_data["page"],
                "text": page_data["text"]
            })

    return all_documents
