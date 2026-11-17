from google import genai
from dotenv import load_dotenv
import os

from src.retriever import retrieve_documents
from src.query_preprocessor import correct_spelling


load_dotenv()


def generate_standalone_question(client, question, chat_history):

    if not chat_history:
        return question

    history_text = ""

    for message in chat_history[-6:]:
        history_text += (
            f"{message['role'].capitalize()}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are a question reformulation assistant.

Your task is to convert the latest user question into
a standalone question using the conversation history.

Rules:
1. Resolve references such as it, they, this, and that.
2. Preserve the user's original meaning.
3. Do not answer the question.
4. Do not add information that was not discussed.
5. Return only the reformulated question.

Conversation history:
{history_text}

Latest user question:
{question}

Standalone question:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text.strip()


def generate_answer(vector_store, question, k=3, chat_history=None):

    if chat_history is None:
        chat_history = []

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Gemini API key not found.")

    client = genai.Client(api_key=api_key)

    standalone_question = generate_standalone_question(
        client,
        question,
        chat_history
    )

    document_vocabulary = getattr(
        vector_store,
        "document_vocabulary",
        set()
    )

    corrected_question = correct_spelling(
        standalone_question,
        document_vocabulary
    )

    retrieved_chunks = retrieve_documents(
        vector_store,
        corrected_question,
        k=k
    )

    if not retrieved_chunks:
        return {
            "answer": (
                "I couldn't find sufficiently relevant information "
                "in the uploaded document to answer your question."
            ),
            "sources": [],
            "corrected_question": corrected_question,
            "standalone_question": standalone_question
        }

    context = ""

    for chunk in retrieved_chunks:
        context += (
            f"\nSource: {chunk['metadata']['source']}\n"
            f"Page: {chunk['metadata']['page']}\n"
            f"Content: {chunk['page_content']}\n"
        )

    history_text = ""

    for message in chat_history[-6:]:
        history_text += (
            f"{message['role'].capitalize()}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are a document question-answering assistant.

Answer the latest question using ONLY the information
provided in the retrieved document context.

Instructions:
1. Answer in simple, clear language.
2. Keep answers concise, ideally 3-5 sentences.
3. Use bullet points only when necessary.
4. Answer using ONLY the retrieved document context.
5. Do not invent facts or use outside knowledge.
6. If the context does not contain the answer,
   clearly say so.
7. Mention the relevant source page number.

Conversation history:
{history_text}

Retrieved document context:
{context}

Original user question:
{question}

Interpreted question:
{corrected_question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return {
        "answer": response.text,
        "sources": retrieved_chunks,
        "corrected_question": corrected_question,
        "standalone_question": standalone_question
    }