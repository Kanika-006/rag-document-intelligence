
from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.rag_pipeline import generate_answer


file_path = "data/sample_documents/sample.pdf"

print("Loading PDF...")
documents = load_pdf(file_path)

print("Splitting document...")
chunks = split_documents(documents)

print("Loading embedding model...")
embedding_model = get_embedding_model()

print("Creating FAISS vector store...")
vector_store = create_vector_store(chunks, embedding_model)

print("\nRAG system is ready!")

while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        print("Exiting RAG system.")
        break

    try:
        result = generate_answer(vector_store, question)

        print("\nGenerated Answer:")
        print(result["answer"])

        print("\nRetrieved Sources:")

        for source in result["sources"]:
            print(
                f"Source: {source['metadata']['source']} | "
                f"Page: {source['metadata']['page']}"
            )

    except Exception as e:
        print("Error:", e)
