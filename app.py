
import os
import hashlib

import streamlit as st

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.rag_pipeline import generate_answer


st.set_page_config(
    page_title="Document Intelligence",
    page_icon="",
    layout="wide"
)


# Initialize session state

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# Application heading

st.title("RAG-Based Document Intelligence System")

st.write(
    "Upload a PDF and ask questions about its content. "
    "The system retrieves relevant passages and generates "
    "answers grounded in your document."
)

st.divider()


# Sidebar

with st.sidebar:

    st.header("Document")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if st.session_state.document_name:
        st.success(
            f"Loaded: {st.session_state.document_name}"
        )

    st.divider()

    st.subheader("Conversation")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Remove document", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.document_hash = None
        st.session_state.document_name = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.caption(
        "Powered by FAISS, BM25, Cross-Encoder reranking "
        "and Google Gemini."
    )


# Process uploaded PDF

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    if current_hash != st.session_state.document_hash:

        try:

            with st.spinner("Processing your PDF..."):

                os.makedirs("temp", exist_ok=True)

                temp_path = os.path.join(
                    "temp",
                    f"{current_hash}.pdf"
                )

                with open(temp_path, "wb") as file:
                    file.write(file_bytes)

                documents = load_pdf(temp_path)

                chunks = split_documents(documents)

                embedding_model = get_embedding_model()

                vector_store = create_vector_store(
                    chunks,
                    embedding_model
                )

                st.session_state.vector_store = vector_store
                st.session_state.document_hash = current_hash
                st.session_state.document_name = uploaded_file.name

                st.session_state.messages = []
                st.session_state.chat_history = []

                st.session_state.total_pages = len(documents)
                st.session_state.total_chunks = len(chunks)

            st.success("PDF processed successfully!")

        except Exception as error:

            st.error(f"Unable to process PDF: {error}")

            st.session_state.vector_store = None
            st.session_state.document_hash = None
            st.session_state.document_name = None


# Document information

if st.session_state.vector_store is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total pages",
            st.session_state.total_pages
        )

    with col2:
        st.metric(
            "Text chunks",
            st.session_state.total_chunks
        )

    st.divider()

    st.subheader("Ask questions about your document")

    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):

                with st.expander("View source references"):

                    for source in message["sources"]:

                        st.write(
                            f"**Document:** "
                            f"{source['metadata']['source']}"
                        )

                        st.write(
                            f"**Page:** "
                            f"{source['metadata']['page']}"
                        )

                        st.divider()

    # User input

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                with st.spinner("Searching and generating answer..."):

                    result = generate_answer(
                        st.session_state.vector_store,
                        question,
                        k=3,
                        chat_history=st.session_state.chat_history
                    )

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:

                    with st.expander("View source references"):

                        displayed_sources = set()

                        for source in sources:

                            source_name = source["metadata"]["source"]
                            page_number = source["metadata"]["page"]

                            source_key = (
                                source_name,
                                page_number
                            )

                            if source_key not in displayed_sources:

                                st.write(
                                    f"**Document:** {source_name}"
                                )

                                st.write(
                                    f"**Page:** {page_number}"
                                )

                                st.divider()

                                displayed_sources.add(source_key)

                else:
                    st.caption(
                        "No sufficiently relevant passages were found."
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

                st.session_state.chat_history.extend([
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ])

                st.session_state.chat_history = (
                    st.session_state.chat_history[-6:]
                )

            except Exception as error:

                st.error(
                    f"Unable to generate an answer: {error}"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry, an error occurred while processing your question."
                })

    st.divider()

    st.caption(
        "Answers are generated using the uploaded document. "
        "Check the source references to verify information."
    )

else:

    st.info(
        "Upload a PDF document from the sidebar to start."
    )
