
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=1000, chunk_overlap=200):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for document in documents:

        text = document["page_content"]

        split_texts = splitter.split_text(text)

        for chunk in split_texts:
            chunks.append({
                "page_content": chunk,
                "metadata": document["metadata"].copy()
            })

    for chunk_id, chunk in enumerate(chunks):
        chunk["metadata"]["chunk_id"] = chunk_id

    return chunks
