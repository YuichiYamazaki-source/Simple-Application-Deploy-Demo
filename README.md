# Tokyo Tourism Guide RAG

A simple RAG (Retrieval-Augmented Generation) prototype for Azure deployment hands-on.

## Overview

Ask questions about Tokyo sightseeing spots, and get AI-generated answers based on local tourism data.

## Tech Stack

- **Backend**: FastAPI (Python)
- **Embedding**: OpenAI Embeddings API (REST, no SDK)
- **Search**: Cosine similarity with NumPy
- **Generation**: OpenAI Chat Completions API (REST, no SDK)
- **Data**: 10 Tokyo sightseeing spots (JSON)

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` file:

```
OPENAI_API_KEY=sk-your-key-here
```

## Run

```bash
uvicorn app:app --reload
```

Open http://localhost:8000 in your browser.

## Architecture

```
User Question
    ↓
Embed (OpenAI API)
    ↓
Cosine Similarity Search (top 3)
    ↓
Build Prompt (question + context)
    ↓
Generate Answer (OpenAI API)
    ↓
Display Response
```

## Project Structure

```
├── app.py              # FastAPI main app
├── rag.py              # RAG logic (embed, search, generate)
├── data.json           # Dummy tourism data
├── templates/
│   └── index.html      # Chat UI
├── requirements.txt
├── .env                # API key (not committed)
└── .env.example        # Template for .env
```
