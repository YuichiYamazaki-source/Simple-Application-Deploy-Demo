import json
import os
import requests
import numpy as np


def load_data(path="data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding(text, api_key):
    """Get embedding vector from OpenAI API (no SDK)."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text,
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def get_embeddings_batch(texts, api_key):
    """Get embeddings for multiple texts in one API call."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": texts,
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()["data"]
    # Sort by index to maintain order
    data.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data]


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query_embedding, doc_embeddings, documents, top_k=3):
    """Find top-k most similar documents."""
    scores = []
    for i, doc_emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((score, i))
    scores.sort(reverse=True)
    results = []
    for score, idx in scores[:top_k]:
        results.append({"document": documents[idx], "score": float(score)})
    return results


def build_context(results):
    """Build context string from search results."""
    parts = []
    for r in results:
        doc = r["document"]
        parts.append(
            f"【{doc['name']} ({doc['name_ja']})】\n"
            f"Area: {doc['area']}\n"
            f"Category: {doc['category']}\n"
            f"Description: {doc['description']}\n"
            f"Highlights: {', '.join(doc['highlights'])}\n"
            f"Access: {doc['access']}\n"
            f"Hours: {doc['hours']}\n"
            f"Admission: {doc['admission']}\n"
            f"Recommended time: {doc['recommended_time']}"
        )
    return "\n\n".join(parts)


def generate_answer(question, context, api_key):
    """Generate answer using OpenAI Chat Completions API (no SDK)."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful Tokyo tourism guide assistant. "
                    "Answer the user's question based on the following sightseeing information. "
                    "If the information doesn't cover the question, say so honestly. "
                    "Answer in the same language as the user's question.\n\n"
                    f"=== Sightseeing Information ===\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.7,
        "max_tokens": 500,
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def make_doc_text(doc):
    """Convert a document to a text string for embedding."""
    return (
        f"{doc['name']} {doc['name_ja']} {doc['area']} {doc['category']} "
        f"{doc['description']} {' '.join(doc['highlights'])}"
    )
