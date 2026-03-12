import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rag import (
    load_data,
    get_embedding,
    get_embeddings_batch,
    search,
    build_context,
    generate_answer,
    make_doc_text,
)

env_path = Path(__file__).parent / ".env"
env_example_path = Path(__file__).parent / ".env.example"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv(env_example_path)

app = FastAPI(title="Tokyo Tourism Guide RAG")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# --- Startup: load data and pre-compute embeddings ---
API_KEY = os.getenv("OPENAI_API_KEY", "")
documents = load_data(str(Path(__file__).parent / "data.json"))
doc_embeddings = []  # populated on first request (lazy init)


def ensure_embeddings():
    """Compute document embeddings once (lazy)."""
    global doc_embeddings
    if doc_embeddings:
        return
    texts = [make_doc_text(doc) for doc in documents]
    doc_embeddings = get_embeddings_batch(texts, API_KEY)


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    ensure_embeddings()

    # 1. Embed the question
    query_emb = get_embedding(body.question, API_KEY)

    # 2. Search similar documents
    results = search(query_emb, doc_embeddings, documents, top_k=3)

    # 3. Build context and generate answer
    context = build_context(results)
    answer = generate_answer(body.question, context, API_KEY)

    sources = [
        {
            "name": r["document"]["name"],
            "name_ja": r["document"]["name_ja"],
            "score": round(r["score"], 4),
        }
        for r in results
    ]

    return AskResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
