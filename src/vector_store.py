import re

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def attach_retrieval_indexes(vector_store):

    documents = list(vector_store.docstore._dict.values())

    tokenized_documents = [
        tokenize(document.page_content)
        for document in documents
    ]

    vector_store.hybrid_documents = documents
    vector_store.bm25_index = BM25Okapi(tokenized_documents)

    document_vocabulary = set()

    for document in documents:
        document_vocabulary.update(
            tokenize(document.page_content)
        )

    vector_store.document_vocabulary = document_vocabulary

    return vector_store


def create_vector_store(chunks, embedding_model):

    if not chunks:
        raise ValueError("No text chunks provided.")

    texts = [
        chunk["page_content"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    return attach_retrieval_indexes(vector_store)


def save_vector_store(vector_store, save_path):
    vector_store.save_local(save_path)


def load_vector_store(save_path, embedding_model):

    vector_store = FAISS.load_local(
        save_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return attach_retrieval_indexes(vector_store)