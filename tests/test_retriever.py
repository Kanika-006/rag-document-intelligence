
from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import retrieve_documents

file_path = "data/sample_documents/sample.pdf"

documents = load_pdf(file_path)

chunks = split_documents(documents)

embedding_model = get_embedding_model()

vector_store = create_vector_store(chunks, embedding_model)

query = input("Enter your question: ")

results = retrieve_documents(vector_store, query, k=3)

print("\nTop 3 Retrieved Results:\n")

for i, result in enumerate(results, start=1):

    print(f"Result {i}")
    print("Source:", result["metadata"]["source"])
    print("Page:", result["metadata"]["page"])
    print("Distance Score:", result["score"])
    print("Content:", result["page_content"])
    print("-" * 60)
