
from src.pdf_loader import load_pdf
from src.text_splitter import split_documents

file_path = "data/sample_documents/sample.pdf"

documents = load_pdf(file_path)

chunks = split_documents(documents)

print("Total PDF pages:", len(documents))
print("Total text chunks:", len(chunks))

print("\nFirst 3 chunks:")

for chunk in chunks[:3]:
    print("\nChunk ID:", chunk["metadata"]["chunk_id"])
    print("Page:", chunk["metadata"]["page"])
    print("Source:", chunk["metadata"]["source"])
    print("Chunk length:", len(chunk["page_content"]))
    print("Text:", chunk["page_content"][:300])
