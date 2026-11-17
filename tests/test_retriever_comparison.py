
from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store

from src.retriever import (
    retrieve_faiss,
    retrieve_hybrid,
    retrieve_documents
)


def main():

    file_path = "data/sample_documents/sample.pdf"

    documents = load_pdf(file_path)
    chunks = split_documents(documents)

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    query = "What is a perceptron?"

    methods = {
        "FAISS Only": retrieve_faiss,
        "Hybrid Search": retrieve_hybrid,
        "Hybrid + Reranking": retrieve_documents
    }

    for method_name, method in methods.items():

        print("\n" + "=" * 55)
        print(method_name)
        print("=" * 55)

        results = method(
            vector_store,
            query,
            k=3,
            candidate_k=10
        )

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print(f"Page: {result['metadata']['page']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['page_content'][:250]}")


if __name__ == "__main__":
    main()
