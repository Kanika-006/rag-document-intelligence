
from src.embeddings import get_embedding_model

model = get_embedding_model()

texts = [
    "How can I reset my password?",
    "I forgot my password and want to change it.",
    "What is artificial intelligence?"
]

embeddings = model.embed_documents(texts)

print("Total sentences:", len(texts))
print("Embedding dimensions:", len(embeddings[0]))
print("First embedding vector:")
print(embeddings[0][:10])
