# RAG Pipeline & Vector Store Specification

This document details how the Flask source code was cloned, parsed, embedded, and stored for retrieval-augmented generation.

---

## Overview

```text
[Flask GitHub Repo]
       │
       ▼
[AST Parsing — chunked by function/class/module]
       │
       ▼
[Embedding via gemini-embedding-001]
       │
       ▼
[ChromaDB Persistent Collection: flask_code (411 chunks)]
       │
       ▼ (at query time)
[User Query → Embed → Query ChromaDB → Top-5 Context → Gemini LLM → Answer]
```

---

## Step 1: Source Acquisition

The Flask framework repository was cloned from `https://github.com/pallets/flask`. Only the core `src/flask/` directory was processed — tests, examples, and documentation were excluded.

- **Repository**: [pallets/flask](https://github.com/pallets/flask)
- **Source directory**: `src/flask/`
- **Files processed**: All `.py` files recursively

---

## Step 2: AST Parsing & Chunking

Each Python file was parsed using Python's `ast` module to produce structured, semantically meaningful chunks:

| Chunk Type | Description | Example |
| :--- | :--- | :--- |
| **Module** | File-level docstring and imports | `flask/app.py` — top-level module docs |
| **Class** | Class definition + docstring + all methods | `class Flask` — application class |
| **Function** | Top-level function + docstring | `add_url_rule()` — route registration |
| **Method** | Class method + docstring | `Flask.run()` — development server |

Each chunk stores:
- `text`: The full source code of the chunk (docstring + signature + body)
- `file`: Relative file path (e.g., `src/flask/app.py`)
- `name`: Symbol name (e.g., `Flask.run`, `add_url_rule`)
- `type`: One of `module`, `class`, `function`, `method`

**Total chunks**: 411

---

## Step 3: Embedding Model

**Model**: `gemini-embedding-001` (Google Gemini Embeddings)

- **Dimensions**: 768
- **Task type**: `RETRIEVAL_QUERY` at query time, `RETRIEVAL_DOCUMENT` at indexing time
- **Batching**: Chunks are embedded in batches of 20 with a 500ms delay between batches to respect rate limits
- **Retry**: Exponential backoff (2^attempt seconds, max 30s) on 429 errors, up to 6 retries

---

## Step 4: Vector Store — ChromaDB

**Database**: ChromaDB (PersistentClient)

- **Location**: `chroma_db/` in the project root
- **Collection name**: `flask_code`
- **Distance metric**: Cosine similarity (ChromaDB default)
- **Contents**: 411 documents, each with:
  - `id`: Auto-generated UUID
  - `embedding`: 768-dimensional float vector
  - `document`: The source code text
  - `metadata`: `{ file, name, type }`

The vector store is pre-built and committed to the repository. It is NOT rebuilt at application startup.

---

## Step 5: Retrieval (Query Time)

When a user asks a question:

1. The query is embedded using `gemini-embedding-001` with task type `RETRIEVAL_QUERY`.
2. ChromaDB performs cosine similarity search against all 411 stored embeddings.
3. The top-5 most similar chunks are returned, each with:
   - Source code text
   - File path
   - Symbol name
   - Distance score
4. Retrieved chunks are formatted into a context block:
   ```
   [src/flask/app.py :: Flask.run]
   def run(self, host=None, port=None, debug=None, ...):
       """Runs the application on a local development server."""
       ...
   ```
5. The context block is injected into the LLM prompt alongside conversation history and the system prompt.

---

## Step 6: Generation Model

**Model**: `gemini-flash-latest` (resolves to `gemini-3.6-flash`)

- **Provider**: Google Generative AI (Gemini API)
- **Capabilities**: Text generation, streaming, function calling
- **System prompt**: Instructs the model to answer naturally, cite file paths and function names inline, refuse to answer from pre-training knowledge if context is insufficient, and use Markdown formatting.

---

## Prompt Structure

```
[System Prompt — RAG behaviour rules]

Conversation so far:
User: previous question
Assistant: previous answer

Context:
[src/flask/app.py :: Flask.add_url_rule]
source code here...

[src/flask/routing.py :: Map.bind]
source code here...

Question: How does Flask handle URL routing?

Answer:
```

- **History window**: Last 6 turns (3 user + 3 assistant) are included.
- **Fallback**: If the model cannot answer from context, it responds: *"I don't have enough source code coverage to answer that."*
