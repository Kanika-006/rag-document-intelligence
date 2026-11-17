
from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(str(file_path))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append({
            "page_content": text.strip(),
            "metadata": {
                "source": file_path.name,
                "page": page_number
            }
        })

    if not documents:
        raise ValueError("No extractable text found in the PDF.")

    return documents
