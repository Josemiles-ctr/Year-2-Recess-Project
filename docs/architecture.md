# System Architecture & Clean Architecture Specification

This document details the architectural layout of the **Flask Framework Expert Assistant — RAG Chatbot**. We employ Uncle Bob's **Clean Architecture** (Onion Architecture) to decouple core business rules from external frameworks (Flask), vector databases, and LLM API clients.

---

## Architecture Overview

The system is organized into concentric circles representing different levels of software abstraction. Dependencies point only **inwards**; the core business domain has no knowledge of Flask, ChromaDB, or the Gemini API.

```text
    ▲  [External Frameworks & Drivers]  (Flask, ChromaDB, Gemini API)
    │           │
    │  [Interface Adapters]             (Controllers, LLM Gateways, Session Presenters)
    │           │
    │  [Use Cases]                      (ChatWithAssistant, ManageSessions)
    │           │
    ▼  [Domain Entities]                (ChatMessage, Session)
```

### 1. Domain Layer (src/domain/)
The innermost circle containing the enterprise business entities — plain Python classes with data and basic validations.
- **ChatMessage**: Holds a single chat turn (`role`: user/assistant, `content`: message text).

### 2. Use Cases Layer (src/interfaces/)
Contains application-specific business rules. Use cases coordinate the flow of data to and from entities.
- **LlmServiceGateway**: Abstract interface defining how the app interacts with an LLM service (`generate`, `generate_stream`, `generate_title`).

### 3. Interface Adapters (src/infrastructure/)
Translates data between the format most convenient for use cases and the format most convenient for external systems.
- **GeminiRagService**: Concrete implementation of `LlmServiceGateway` that queries ChromaDB for relevant Flask source code chunks and calls the Gemini API.
- **SessionStore**: In-memory session storage for chat history, titles, and session metadata.
- **Web Routes (Flask Blueprint)**: Adapts HTTP requests into use case invocations and converts responses back to JSON or rendered templates.

### 4. Frameworks & Drivers
The outermost circle, composed of frameworks and tools.
- **Flask**: Web server, Blueprint routing, Jinja templating, session cookies.
- **ChromaDB**: Persistent vector database storing 411 AST-parsed Flask source code chunks.
- **Gemini API**: Google's generative AI model for embeddings and text generation.

---

## Data Flow (User Query → Response)

1. **Request Entry**: User types a question in the chat UI. The request hits the Flask route `POST /api/chat`.
2. **Session Lookup**: The route loads the chat history from `SessionStore` for the current session.
3. **RAG Retrieval**: `GeminiRagService.retrieve()` embeds the query via `gemini-embedding-001` and queries ChromaDB for the top-5 most similar Flask source chunks.
4. **Prompt Assembly**: The retrieved chunks, conversation history (last 6 turns), and system prompt are compiled into a single prompt.
5. **LLM Generation**: The prompt is sent to `gemini-flash-latest`. The response is collected (or streamed).
6. **Persistence**: The user message and assistant response are appended to the session history in `SessionStore`.
7. **Title Generation**: If this is the session's first message, `GeminiRagService.generate_title()` creates a concise title.
8. **Response**: The response JSON is returned to the browser and rendered as Markdown.

---

## Key Design Patterns Used

- **Dependency Injection**: Dependencies are injected via Flask's `app.config` (e.g., `RAG_SERVICE`, `SESSION_STORE`), making unit testing straightforward.
- **Boundary Interfaces**: `LlmServiceGateway` (abstract base class) establishes a boundary between use cases and external LLM providers. Switching from Gemini to another provider only requires a new infrastructure adapter.
- **Repository Pattern**: `SessionStore` abstracts session persistence behind a simple dict-like interface. Can be swapped for a database-backed store without changing use case code.
- **Factory Pattern**: `create_app()` in `app_setup.py` wires all dependencies together, loads environment variables, and returns a configured Flask instance.
