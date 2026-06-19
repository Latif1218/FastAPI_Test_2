<div align="center">

# Local OCR & Dynamic RAG System

**A fully local, multilingual (বাংলা + English) document intelligence pipeline.**
OCR → Chunking → Embedding → Hybrid Vector Search → Local LLM Answers — with zero external API calls.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-9b59b6)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%2F%20Llama3-000000)](https://ollama.com/)
[![Tesseract](https://img.shields.io/badge/OCR-Tesseract-43a047)](https://github.com/tesseract-ocr/tesseract)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

</div>

---

## Overview

This project ingests scanned documents and PDFs containing **Bangla, English, or mixed text**, extracts the text entirely on-device using **Tesseract OCR**, and builds a searchable knowledge base using a **hybrid Retrieval-Augmented Generation (RAG)** pipeline. Users can ask natural-language questions and apply **manual metadata filters** (language, document type, date range, filename) alongside semantic search — all powered by a **local LLM (Ollama)**, so no document content or query ever leaves the machine.

A bilingual (বাং / EN) single-file web frontend is included for uploading documents and running searches without touching the API docs.

---

## Table of Contents

- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Option A — Docker](#option-a--docker-recommended)
  - [Option B — Local (Windows / macOS / Linux)](#option-b--local-setup)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Architecture

```
                 Upload PDF / Image
                         │
                         ▼
        OCR — Tesseract (ben+eng), 100% local
                         │
                         ▼
       Text Extraction + Language Detection
                         │
                         ▼
      Chunking — word-based, 500 words / 100 overlap
                         │
                         ▼
     Embedding — intfloat/multilingual-e5-base
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│   ChromaDB — vectors + chunk text + metadata     │
│   (persisted locally, no external service)       │
└─────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
    Natural Language Query   Manual Metadata Filter
            │                (language / type / date / filename)
            └────────────┬────────────┘
                         ▼
        Hybrid Search — metadata `where` filter
        narrows candidates, THEN cosine similarity
        search runs on the filtered subset
                         │
                         ▼
              Top-K Context Retrieval
                         │
                         ▼
           Local LLM — Ollama (Llama 3)
                         │
                         ▼
        Final Answer + cited source chunks
```

---

## Design Decisions

### OCR — Tesseract (`ben+eng`)
Open-source and runs fully offline, with a maintained Bangla language pack. **Trade-off:** accuracy on complex/connected Bangla conjuncts (যুক্তাক্ষর) and low-quality scans is noticeably lower than commercial OCR — expect roughly **80–90% character accuracy** on clean printed Bangla text, dropping further on handwriting or noisy scans. The OCR layer is isolated in `app/core/ocr.py`, so it can be swapped for a vision-language OCR model (e.g. Surya) without touching the rest of the pipeline.

### Chunking — word-based, 500 words / 100-word overlap
Word-based splitting (instead of fixed character count) keeps "semantic size" consistent across Bangla and English, since the two scripts differ in characters-per-word. The overlap ensures ideas spanning a chunk boundary aren't lost from context — see `app/core/chunking.py`.

### Embedding — `intfloat/multilingual-e5-base`
Trained across 100+ languages including Bangla, with strong cross-lingual retrieval performance. Uses the model's native `query:` / `passage:` instruction prefixes to improve retrieval quality — see `app/core/embedding.py`.

### Vector Store — ChromaDB
Chosen over FAISS because it stores **vectors, raw text, and metadata together** and supports native `where`-clause filtering. This directly powers the hybrid search requirement: metadata filters are applied first to narrow the candidate set, then cosine similarity search runs only within that filtered subset. See `build_where_clause()` and `query_chunks()` in `app/core/vector_store.py`.

### LLM — Ollama (Llama 3), local
Keeps the entire pipeline on-machine with zero external API calls — see `app/core/rag.py`.

---

## Project Structure

```
local-ocr-rag/
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── config.py               # Centralized settings (.env driven)
│   ├── api/
│   │   ├── upload.py           # POST /upload
│   │   └── search.py           # POST /search
│   ├── core/
│   │   ├── ocr.py               # Tesseract OCR + language detection
│   │   ├── chunking.py          # Word-based chunking with overlap
│   │   ├── embedding.py         # multilingual-e5 embedding wrapper
│   │   ├── vector_store.py      # ChromaDB hybrid search logic
│   │   └── rag.py               # Ollama LLM call with retrieved context
│   └── schemas/
│       └── schemas.py           # Pydantic request/response models
├── frontend/
│   └── index.html               # Bilingual (বাং/EN) single-file web UI
├── storage/
│   ├── uploads/                  # Raw uploaded files
│   └── chroma_db/                 # Persisted ChromaDB data
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

---

## Setup

### Option A — Docker (recommended)

```bash
docker-compose up --build
```

Pull the LLM model inside the Ollama container (first time only):

```bash
docker exec -it ollama ollama pull llama3
```

The API will be available at `http://127.0.0.1:8000` (interactive docs at `/docs`).

### Option B — Local Setup

#### 1. Install system dependencies

**Tesseract OCR + Bangla language pack**

| OS | Command |
|---|---|
| Ubuntu/Debian | `sudo apt install tesseract-ocr tesseract-ocr-ben poppler-utils` |
| macOS | `brew install tesseract tesseract-lang poppler` |
| Windows | Install via the [UB-Mannheim Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki), check **Bengali** under additional language data during setup |

> **Windows users:** after installing, set the path in `.env`:
> ```env
> TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
> ```
> Verify with:
> ```powershell
> & "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
> ```

**Ollama (local LLM runtime)**

Download and install from [ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull llama3
```

> Low-RAM machines (≤8GB) can use a smaller model instead:
> ```bash
> ollama pull llama3.2:1b
> ```
> Then set `OLLAMA_MODEL=llama3.2:1b` in `.env`.

> **Windows note:** if `ollama` isn't recognized after install, open a **new** terminal window (PATH refreshes per-session). If it still fails inside an activated virtual environment, call it by full path:
> ```powershell
> & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3
> ```

#### 2. Set up the Python environment

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

#### 3. Configure environment variables

Copy and adjust `.env` as needed (defaults work out of the box for most setups):

```env
OCR_LANGUAGES=ben+eng
CHUNK_SIZE=500
CHUNK_OVERLAP=100
EMBEDDING_MODEL=intfloat/multilingual-e5-base
CHROMA_COLLECTION_NAME=documents
OLLAMA_MODEL=llama3
OLLAMA_HOST=http://localhost:11434
```

---

## Running the App

```bash
uvicorn app.main:app --reload
```

- API root: `http://127.0.0.1:8000`
- Interactive Swagger docs: `http://127.0.0.1:8000/docs`
- Web frontend: open `frontend/index.html` directly in your browser

---

## API Reference

### `POST /upload`

Uploads and processes a document (OCR → chunk → embed → store).

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@/path/to/document.pdf"
```

**Response**

```json
{
  "filename": "document.pdf",
  "doc_id": "f570795c-485f-4813-813d-6c9e1f34a91b",
  "language_detected": "bn",
  "total_chunks": 4,
  "message": "Document processed and stored successfully (fully local pipeline)."
}
```

### `POST /search`

Hybrid semantic + metadata-filtered search with a generated answer.

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
        "query": "এই ডকুমেন্টে কী বলা হয়েছে?",
        "top_k": 5,
        "filters": {
          "language": "bn",
          "doc_type": "pdf",
          "date_from": "2026-01-01"
        }
      }'
```

**Request fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✅ | Natural language question |
| `top_k` | integer | optional (default `5`) | Number of chunks to retrieve |
| `filters.language` | `"bn"` \| `"en"` \| `"bn+en"` | optional | Restrict to a detected language |
| `filters.doc_type` | `"pdf"` \| `"image"` | optional | Restrict to a document type |
| `filters.date_from` / `date_to` | `YYYY-MM-DD` | optional | Restrict by upload date range |
| `filters.filename` | string | optional | Restrict to one exact source file |

**Response**

```json
{
  "query": "এই ডকুমেন্টে কী বলা হয়েছে?",
  "answer": "...",
  "sources": [
    {
      "chunk_id": "f570795c..._chunk_0",
      "text": "...",
      "metadata": { "filename": "document.pdf", "language": "bn", "doc_type": "pdf", "upload_date": "2026-06-19" },
      "score": 0.87
    }
  ]
}
```

---

## Frontend

A single dependency-free HTML file (`frontend/index.html`) provides:

- 📤 **Upload panel** — drag-and-drop or click to upload, with a live processing log (OCR status, detected language, chunk count)
- 🔎 **Search panel** — natural language query box plus manual filters (language, document type, date range, filename)
- 📊 **Answer + sources** — generated answer alongside the retrieved chunks, similarity scores, and metadata
- 🌐 **বাং / EN toggle** — switches the entire interface, including live log messages, between Bangla and English instantly
- 🟢 **Live backend status indicator** — shows whether the FastAPI server is reachable

To use it, just open the file in any browser while the backend is running on `http://localhost:8000` (or update the **API base** field in the UI to match your server address).

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Failed to fetch` in the frontend | Backend not running / crashed | Confirm `uvicorn` is still running; check terminal for tracebacks |
| `ollama` not recognized in PowerShell | PATH not refreshed in current session | Open a new terminal, or call Ollama by full path |
| OCR returns empty/garbled text | Missing Bangla language pack | Reinstall Tesseract with **Bengali** checked, or run `tesseract --list-langs` to confirm `ben` is installed |
| `/search` returns no results | Filters too strict / no matching documents | Remove filters one at a time to isolate the mismatch, or check the `language`/`doc_type` values used at upload time |
| Slow first request | Embedding model and LLM load lazily on first use | Expected — subsequent requests are faster |

---

## License

Apache-2.0 license — free to use, modify, and extend for academic or personal projects.
