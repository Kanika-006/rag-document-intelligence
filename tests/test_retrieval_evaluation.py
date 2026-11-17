
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store

from src.retriever import (
    retrieve_faiss,
    retrieve_hybrid,
    retrieve_documents
)

from evaluation_dataset import evaluation_queries


K = 3
CANDIDATE_K = 20


def get_unique_pages(results):

    pages = []

    for result in results:

        page = int(result["metadata"]["page"])

        if page not in pages:
            pages.append(page)

    return pages[:K]


def calculate_metrics(retrieved_pages, relevant_pages):

    relevant_pages = set(relevant_pages)

    relevant_retrieved = [
        page for page in retrieved_pages
        if page in relevant_pages
    ]

    precision = len(relevant_retrieved) / K

    recall = (
        len(relevant_retrieved) / len(relevant_pages)
        if relevant_pages else 0
    )

    reciprocal_rank = 0

    for rank, page in enumerate(retrieved_pages, start=1):

        if page in relevant_pages:
            reciprocal_rank = 1 / rank
            break

    hit = 1 if relevant_retrieved else 0

    return precision, recall, reciprocal_rank, hit


def evaluate_method(vector_store, retrieval_function, method_name):

    total_precision = 0
    total_recall = 0
    total_mrr = 0
    total_hits = 0

    print(f"\nEvaluating {method_name}...")

    for index, item in enumerate(evaluation_queries, start=1):

        question = item["question"]
        relevant_pages = item["relevant_pages"]

        results = retrieval_function(
            vector_store,
            question,
            k=CANDIDATE_K,
            candidate_k=CANDIDATE_K
        )

        retrieved_pages = get_unique_pages(results)

        precision, recall, mrr, hit = calculate_metrics(
            retrieved_pages,
            relevant_pages
        )

        total_precision += precision
        total_recall += recall
        total_mrr += mrr
        total_hits += hit

        print(
            f"Question {index:02d}: "
            f"Retrieved={retrieved_pages} | "
            f"Relevant={relevant_pages}"
        )

    total_queries = len(evaluation_queries)

    return {
        "Precision@3": total_precision / total_queries,
        "Recall@3": total_recall / total_queries,
        "MRR": total_mrr / total_queries,
        "Hit Rate@3": total_hits / total_queries
    }


def main():

    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION")
    print("=" * 65)

    print("\nLoading PDF...")

    documents = load_pdf(
        "data/sample_documents/sample.pdf"
    )

    print("Splitting document...")

    chunks = split_documents(documents)

    print("Loading embedding model...")

    embedding_model = get_embedding_model()

    print("Building retrieval indexes...")

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    print("\nDocument processed successfully.")
    print(f"Pages: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Evaluation questions: {len(evaluation_queries)}")

    methods = [
        ("FAISS Only", retrieve_faiss),
        ("Hybrid Search", retrieve_hybrid),
        ("Hybrid + CrossEncoder", retrieve_documents)
    ]

    all_results = {}

    for method_name, retrieval_function in methods:

        metrics = evaluate_method(
            vector_store,
            retrieval_function,
            method_name
        )

        all_results[method_name] = metrics

    print("\n" + "=" * 75)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 75)

    print(
        f"\n{'Method':<27}"
        f"{'Precision@3':>12}"
        f"{'Recall@3':>12}"
        f"{'MRR':>10}"
        f"{'Hit Rate@3':>14}"
    )

    print("-" * 75)

    for method_name, metrics in all_results.items():

        print(
            f"{method_name:<27}"
            f"{metrics['Precision@3']:>12.4f}"
            f"{metrics['Recall@3']:>12.4f}"
            f"{metrics['MRR']:>10.4f}"
            f"{metrics['Hit Rate@3']:>14.4f}"
        )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
