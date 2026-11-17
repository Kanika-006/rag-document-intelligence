
from src.pdf_loader import load_pdf

file_path = "data/sample_documents/sample.pdf"

documents = load_pdf(file_path)

print("Total extracted pages:", len(documents))

for document in documents[:3]:
    print("\nPage:", document["metadata"]["page"])
    print("Source:", document["metadata"]["source"])
    print("Extracted text:")
    print(document["page_content"][:500])
