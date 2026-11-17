
from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store, save_vector_store, load_vector_store

file_path = "data/sample_documents/sample.pdf"

documents = load_pdf(file_path)

chunks = split_documents(documents)

embedding_model = get_embedding_model()

vector_store = create_vector_store(chunks, embedding_model)

print("Total chunks stored:", len(chunks))

save_vector_store(vector_store, "vectorstore/faiss_index")

print("Vector store saved successfully.")

loaded_store = load_vector_store(
    "vectorstore/faiss_index",
    embedding_model
)

print("Vector store loaded successfully.")
