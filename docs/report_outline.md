# Academic Submission Report Outline (Group O)

This document provides a detailed structural guide and template for compiling the final 10-page academic reports required for Part A (Data Science & ML) and Part B (Flask Web Development).

---

## Part A: RAG System Design & Evaluation Report Layout
*Limit: 10 Pages Maximum. Recommended formatting: Calibri/Arial 11pt, 1.15 line spacing.*

### Page 1: Title Page & Executive Summary
- **Title**: *Retrieval-Augmented Generation for Source Code Q&A: A Flask Framework Expert Assistant.*
- **Header**: Course BSE2301 Software Engineering Mini Project 2. Group O.
- **Abstract**: Concise overview of the problem (understanding large codebases), our RAG approach, embedding strategy, and evaluation.

### Pages 2-3: Section 1 — Source Acquisition & Chunking Strategy
- **Flask Repository**: Describe cloning `pallets/flask` and selecting `src/flask/` as the corpus.
- **AST Parsing**: Explain how Python's `ast` module was used to split source into module/class/function/method chunks.
- **Chunk Statistics**: Table showing total chunks (411), breakdown by type (module, class, function, method), and average chunk size.
- **Rationale**: Why AST chunking preserves semantic boundaries better than naive sliding-window splitting.

### Pages 4-5: Section 2 — Embedding & Vector Store
- **Embedding Model**: Describe `gemini-embedding-001` (768 dimensions, task-specific RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY).
- **Vector Database**: ChromaDB persistent collection, cosine similarity, batch embedding process with rate-limit handling.
- **Retrieval Strategy**: Top-K selection (K=5), distance metrics, and relevance filtering.

### Pages 6-7: Section 3 — LLM Integration & Prompt Design
- **Generation Model**: `gemini-flash-latest` (Gemini 3.6 Flash), API integration via `google-genai` SDK.
- **System Prompt**: Explain the grounding constraints, citation rules, and refusal behaviour.
- **Context Assembly**: Show the prompt template with conversation history, retrieved context, and user question.
- **Streaming**: SSE implementation for token-by-token response delivery.

### Pages 8-9: Section 4 — Web Application & Session Management
- **Flask Architecture**: Clean Architecture layers (domain → interfaces → infrastructure).
- **Session Store**: In-memory dictionary-based storage with CRUD operations.
- **Frontend**: Responsive chat UI, Markdown rendering, copy buttons, light/dark theme, suggestion chips.
- **API Endpoints**: Table of all REST endpoints with request/response formats.

### Page 10: Section 5 — Conclusions & Recommendations
- **Effectiveness**: Evaluate RAG quality with sample Q&A pairs showing grounded vs. hallucinated responses.
- **Limitations**: In-memory sessions, daily API quota limits, single-model dependency.
- **Recommendations**: Database-backed sessions, multi-model fallback, expanded corpus.

---

## Part B: Flask Web Service Report Layout
*Limit: 10 Pages Maximum.*

### Page 1: Title Page & Web Service Abstract
- **Title**: *RAG-Powered Code Assistant Web Interface using Flask Clean Architecture.*
- **Details**: Github links, supervision logs, and team responsibilities.

### Pages 2-3: Section 1 — Design Architecture
- **Intro**: Describe the chat interface, session management, and RAG pipeline.
- **Architecture Model**: Clean Architecture UML/block diagram from docs/architecture.md. Explain:
  - Domain isolation.
  - Use-case flow control.
  - Dependency inversion (gateways).

### Pages 4-6: Section 2 — Web Application Systems Documentation
- **Core Modules**:
  - app.py (Application factory)
  - routes.py (Controller router mapping)
  - gemini_service.py (RAG + LLM integration)
  - session_store.py (Session persistence)
  - entities.py (Business concepts)
- **RAG Integration Flow**: Document how ChromaDB queries and Gemini API calls are orchestrated.

### Pages 7-9: Section 3 — System Execution Screenshots
*Include high-quality screenshots with labels explaining UI behaviours:*
- **Screenshot 1**: Landing page with welcome message and suggestion chips.
- **Screenshot 2**: Active chat session showing user and assistant messages with Markdown rendering.
- **Screenshot 3**: Session list sidebar with multiple conversations.
- **Screenshot 4**: Light/dark theme toggle in action.

### Page 10: Section 4 — Git Versioning & Supervision Summary
- **Git log representation**: Summary showing active participation of all team members.
- **Supervision logs**: Dates, feedback from lecturers, and final checklist compliance.
