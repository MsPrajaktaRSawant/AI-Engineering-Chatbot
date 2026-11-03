from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):

    # Read PDF document
    reader = PdfReader(str(pdf_path))

    extracted_text = []

    # Extract text from every page
    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():
            extracted_text.append(text)

        print(f"Processed page {page_number}")

    # Combine all extracted pages
    full_text = "\n".join(extracted_text)

    return full_text


if __name__ == "__main__":

    # Locate sample PDF
    project_root = Path(__file__).resolve().parent.parent

    pdf_path = (
        project_root
        / "documents"
        / "ai_engineering_rag_practice_guide.pdf"
    )

    # Extract PDF text
    text = extract_text_from_pdf(pdf_path)

    print("\nPDF EXTRACTION COMPLETED")
    print("=" * 40)

    print("Total characters:", len(text))

    print("\nExtracted text preview:")
    print(text[:2000])
