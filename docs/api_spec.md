# Web API Specification & Documentation

This document describes the endpoints, input parameters, session payloads, and response JSON formats for the Flask Framework Expert Assistant.

---

## Web Interface Routes

### 1. Main Chat Interface
- **Route**: `GET /`
- **Description**: Returns the single-page chat interface. Includes the sidebar (session list, user info, theme toggle) and the main chat area (welcome message, message stream, input form).
- **Query Parameters**: None.
- **Response**: HTML template (index.html) with optional `user_name` from Flask session.

---

## API Endpoints

### 1. Set User Name
Stores the user's display name in the Flask session cookie.

- **Route**: `POST /api/set-name`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  { "name": "Alice" }
  ```
- **Response (200 OK)**:
  ```json
  { "status": "success", "name": "Alice" }
  ```

---

### 2. List Sessions
Returns all sessions for the current user, sorted by creation date (newest first).

- **Route**: `GET /api/sessions`
- **Response (200 OK)**:
  ```json
  {
    "status": "success",
    "sessions": [
      {
        "id": "uuid-string",
        "title": "How Flask routing works",
        "created_at": "2026-07-23T12:00:00Z",
        "message_count": 3
      }
    ]
  }
  ```

---

### 3. Create Session
Creates a new empty chat session.

- **Route**: `POST /api/sessions`
- **Response (200 OK)**:
  ```json
  { "status": "success", "session_id": "uuid-string" }
  ```

---

### 4. Get Session
Returns full session data including chat history.

- **Route**: `GET /api/sessions/<session_id>`
- **Response (200 OK)**:
  ```json
  {
    "status": "success",
    "session": {
      "id": "uuid-string",
      "title": "How Flask routing works",
      "created_at": "2026-07-23T12:00:00Z",
      "history": [
        { "role": "user", "content": "How does Flask handle URL routing?" },
        { "role": "assistant", "content": "Flask uses the `add_url_rule` method..." }
      ]
    }
  }
  ```
- **Error Response (404)**:
  ```json
  { "status": "error", "message": "Session not found" }
  ```

---

### 5. Delete Session
Deletes a chat session and its history.

- **Route**: `DELETE /api/sessions/<session_id>`
- **Response (200 OK)**:
  ```json
  { "status": "success" }
  ```

---

### 6. Update Session Title
Manually updates the session title.

- **Route**: `PUT /api/sessions/<session_id>/title`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  { "title": "New custom title" }
  ```
- **Response (200 OK)**:
  ```json
  { "status": "success" }
  ```

---

### 7. Send Chat Message (Non-Streaming)
Sends a user message and receives the full assistant response.

- **Route**: `POST /api/chat`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "message": "How does Flask handle URL routing?",
    "session_id": "uuid-string"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "status": "success",
    "response": {
      "content": "Flask registers routes via the `add_url_rule` method...\n\nIn `src/flask/app.py`, `Flask.add_url_rule`..."
    }
  }
  ```
- **Error Response (500)**:
  ```json
  { "status": "error", "message": "The AI service is temporarily out of requests due to a quota limit. Please wait about 35 seconds before trying again." }
  ```

---

### 8. Send Chat Message (Streaming via SSE)
Streams the assistant response token-by-token using Server-Sent Events.

- **Route**: `POST /api/chat/stream`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "message": "How does Flask handle URL routing?",
    "session_id": "uuid-string"
  }
  ```
- **Response**: `text/event-stream`
  ```
  data: Flask
  data:  registers
  data:  routes
  data:  via
  data: [DONE]
  ```

---

## Session Management

Sessions are stored server-side in an in-memory `SessionStore`, keyed by user name:
- **Create**: `POST /api/sessions` generates a UUID and initialises an empty history.
- **Read**: `GET /api/sessions/<id>` returns the session with full message history.
- **Update**: Messages are appended via `POST /api/chat` and titles via `PUT /api/sessions/<id>/title`.
- **Delete**: `DELETE /api/sessions/<id>` removes the session.

**Note**: Stored in memory only — sessions are lost on server restart. The user name is persisted in a Flask signed session cookie.
