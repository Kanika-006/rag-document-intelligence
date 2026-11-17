
import re

from sentence_transformers import CrossEncoder


_reranker = None


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def get_reranker():

    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu"
        )

    return _reranker


def get_document_key(document):

    return (
        document.page_content,
        tuple(sorted(document.metadata.items()))
    )


def format_result(document, score, semantic_distance=None, fusion_score=None):

    return {
        "page_content": document.page_content,
        "metadata": document.metadata,
        "score": float(score),
        "semantic_distance": semantic_distance,
        "fusion_score": fusion_score
    }


def retrieve_faiss(
    vector_store,
    query,
    k=3,
    candidate_k=10
):

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    candidate_k = min(candidate_k, vector_store.index.ntotal)

    results = vector_store.similarity_search_with_score(
        query,
        k=candidate_k
    )

    retrieved = []

    for document, distance in results[:k]:

        retrieved.append(
            format_result(
                document,
                score=-float(distance),
                semantic_distance=float(distance)
            )
        )

    return retrieved


def retrieve_hybrid(
    vector_store,
    query,
    k=3,
    candidate_k=10
):

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not hasattr(vector_store, "bm25_index"):
        raise ValueError("BM25 index not found. Rebuild the vector store.")

    candidate_k = min(candidate_k, vector_store.index.ntotal)

    candidates = {}
    semantic_distances = {}
    fusion_scores = {}

    # 1. Semantic retrieval using FAISS

    semantic_results = vector_store.similarity_search_with_score(
        query,
        k=candidate_k
    )

    for rank, (document, distance) in enumerate(
        semantic_results,
        start=1
    ):

        key = get_document_key(document)

        candidates[key] = document
        semantic_distances[key] = float(distance)

        fusion_scores[key] = (
            fusion_scores.get(key, 0)
            + 1 / (60 + rank)
        )

    # 2. Keyword retrieval using BM25

    tokenized_query = tokenize(query)

    bm25_scores = vector_store.bm25_index.get_scores(
        tokenized_query
    )

    ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda index: bm25_scores[index],
        reverse=True
    )

    bm25_rank = 0

    for index in ranked_indices:

        if bm25_scores[index] <= 0:
            continue

        bm25_rank += 1

        if bm25_rank > candidate_k:
            break

        document = vector_store.hybrid_documents[index]
        key = get_document_key(document)

        candidates[key] = document

        fusion_scores[key] = (
            fusion_scores.get(key, 0)
            + 1 / (60 + bm25_rank)
        )

    # 3. Reciprocal Rank Fusion

    fused_candidates = sorted(
        candidates.items(),
        key=lambda item: fusion_scores[item[0]],
        reverse=True
    )

    retrieved = []

    for key, document in fused_candidates[:k]:

        retrieved.append(
            format_result(
                document,
                score=fusion_scores[key],
                semantic_distance=semantic_distances.get(key),
                fusion_score=fusion_scores[key]
            )
        )

    return retrieved


def retrieve_documents(
    vector_store,
    query,
    k=3,
    candidate_k=10,
    min_reranker_score=None
):

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # Get hybrid candidates before reranking

    hybrid_candidates = retrieve_hybrid(
        vector_store,
        query,
        k=candidate_k,
        candidate_k=candidate_k
    )

    if not hybrid_candidates:
        return []

    # CrossEncoder reranking

    reranker = get_reranker()

    pairs = [
        (query, result["page_content"])
        for result in hybrid_candidates
    ]

    reranker_scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    reranked_results = []

    for result, reranker_score in zip(
        hybrid_candidates,
        reranker_scores
    ):

        reranker_score = float(reranker_score)

        if (
            min_reranker_score is not None
            and reranker_score < min_reranker_score
        ):
            continue

        result["score"] = reranker_score
        result["reranker_score"] = reranker_score

        reranked_results.append(result)

    reranked_results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True
    )

    return reranked_results[:k]
