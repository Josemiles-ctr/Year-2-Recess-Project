# Web API Specification & Documentation

This document describes the endpoints, input parameters, session payloads, and response JSON formats for the AuraScan Flask web service.

---

## Web Interface Routes

### 1. Landing Page
- **Route**: `GET /`
- **Description**: Public landing page. If the user is authenticated and has existing scan sessions, redirects to the report dashboard (`/report?sid=<latest>`). If authenticated with no sessions, redirects to the upload page (`/upload`). Otherwise renders the `index.html` template.
- **Response**: HTML template or redirect.

### 2. Scan Upload Page
- **Route**: `GET /upload`
- **Authentication**: Required (`@login_required`)
- **Description**: Presents the file upload dropzone for chest X-ray images. Passes `has_sessions` flag to template for conditional "View Reports" button.
- **Response**: HTML template (`upload.html`).

### 3. Report Dashboard
- **Route**: `GET /report`
- **Query Parameters**:
  | Key | Type | Required | Description |
  | :--- | :--- | :--- | :--- |
  | `sid` | Integer | No | Session ID to view. If omitted, loads the most recent session. |
- **Authentication**: Required (`@login_required`)
- **Description**: Serves the detailed diagnostic report page with model predictions side-by-side, AI narrative summary, and the interactive chat assistant. Acts as the main dashboard after login.
- **Response**: HTML template (`report.html`).

---

## API Endpoints

### 1. Analyze X-Ray Scan
Processes the uploaded image, triggers the dual-model classification engines (Random Forest + CNN), generates an LLM narrative, and persists the session to the database.

- **Route**: `POST /api/analyze`
- **Authentication**: Required (`@login_required`)
- **Content-Type**: `multipart/form-data`
- **Request Parameters**:
  | Key | Type | Required | Description |
  | :--- | :--- | :--- | :--- |
  | `file` | Binary | Yes | The chest X-ray image (PNG, JPG, JPEG). |
  | `note` | String | No | Optional review note for the scan. |

- **Example Response (200 OK)**:
  ```json
  {
    "status": "success",
    "session_id": 42,
    "filename": "patient_xray.png",
    "timestamp": "2026-07-20T12:50:00Z",
    "traditional_model": {
      "prediction": "Malignant",
      "confidence": 0.824,
      "per_class_probabilities": {
        "Atelectasis": 0.02,
        "Consolidation": 0.01,
        "Infiltration": 0.03,
        "Mass": 0.82,
        "Nodule": 0.05,
        "Pneumothorax": 0.01,
        "Normal": 0.06
      }
    },
    "cnn_model": {
      "prediction": "Malignant",
      "confidence": 0.895,
      "per_class_probabilities": {
        "Atelectasis": 0.01,
        "Consolidation": 0.02,
        "Infiltration": 0.02,
        "Mass": 0.89,
        "Nodule": 0.03,
        "Pneumothorax": 0.01,
        "Normal": 0.02
      }
    },
    "llm_narrative": "<h3>Summary of Findings</h3><p>The analysis indicates a malignant mass...</p>"
  }
  ```

- **Error Response (400 Bad Request)**:
  ```json
  {
    "status": "error",
    "message": "No file uploaded."
  }
  ```

---

### 2. Chat with AI Assistant
Sends a follow-up question to the LLM with the full diagnostic context from the session.

- **Route**: `POST /api/chat`
- **Authentication**: Required (`@login_required`)
- **Content-Type**: `application/json`
- **Request Payload**:
  ```json
  {
    "message": "Why did the traditional classifier flag this image as malignant?",
    "session_id": 42
  }
  ```

- **Example Response (200 OK)**:
  ```json
  {
    "status": "success",
    "response": {
      "role": "assistant",
      "content": "The traditional classifier labeled this scan as malignant primarily due to high image contrast (18.23) and low homogeneity (0.35) inside the tissue matrix...",
      "timestamp": "2026-07-20T12:55:00Z"
    }
  }
  ```

- **Error Response (400 Bad Request)**:
  ```json
  {
    "status": "error",
    "message": "session_id required."
  }
  ```

---

### 3. Clear Chat History
Deletes all chat messages for a given session.

- **Route**: `POST /api/chat/clear`
- **Authentication**: Required (`@login_required`)
- **Content-Type**: `application/json`
- **Request Payload**:
  ```json
  {
    "session_id": 42
  }
  ```
- **Example Response (200 OK)**:
  ```json
  {
    "status": "success",
    "message": "Chat cleared."
  }
  ```

---

### 4. List Sessions
Returns all scan sessions for the authenticated user.

- **Route**: `GET /api/sessions`
- **Authentication**: Required (`@login_required`)
- **Example Response (200 OK)**:
  ```json
  {
    "status": "success",
    "sessions": [
      {
        "id": 42,
        "title": "Malignant mass detected in upper left lobe",
        "scan_filename": "patient_xray.png",
        "created_at": "2026-07-20T12:50:00+00:00",
        "message_count": 5
      }
    ]
  }
  ```

---

### 5. Get Session Detail
Returns full session data including chat history and scan results.

- **Route**: `GET /api/sessions/<int:sid>`
- **Authentication**: Required (`@login_required`)
- **Example Response (200 OK)**:
  ```json
  {
    "status": "success",
    "session": {
      "id": 42,
      "title": "Malignant mass detected in upper left lobe",
      "scan_filename": "patient_xray.png",
      "history": [
        {"role": "user", "content": "What is the primary finding?"},
        {"role": "assistant", "content": "The primary finding indicates a malignant mass..."}
      ],
      "scan_data": {
        "traditional_prediction": "Malignant",
        "traditional_confidence": 0.824,
        "cnn_prediction": "Malignant",
        "cnn_confidence": 0.895,
        "llm_narrative": "<h3>Summary</h3>..."
      }
    }
  }
  ```

---

### 6. Delete Session
Permanently removes a scan session and all its associated messages.

- **Route**: `DELETE /api/sessions/<int:sid>`
- **Authentication**: Required (`@login_required`)
- **Example Response (200 OK)**:
  ```json
  {
    "status": "success"
  }
  ```

---

### 7. Update Session Title
Updates the display title of a scan session.

- **Route**: `PUT /api/sessions/<int:sid>/title`
- **Authentication**: Required (`@login_required`)
- **Content-Type**: `application/json`
- **Request Payload**:
  ```json
  {
    "title": "Updated case description"
  }
  ```
- **Example Response (200 OK)**:
  ```json
  {
    "status": "success"
  }
  ```

---

## Authentication

All endpoints except `GET /` require the user to be authenticated. Authentication uses **Flask-Login** with session-based cookies. Unauthenticated requests to protected routes are redirected to `GET /login`.

### Auth Endpoints

#### Login
- **Route**: `GET/POST /login`
- **POST Parameters**: `email`, `password`
- **On success**: Redirects to `/report` (dashboard) or the originally requested page.

#### Register
- **Route**: `GET/POST /register`
- **POST Parameters**: `email`, `password`, `confirm`
- **On success**: Auto-logs in and redirects to `/report` (dashboard).

#### Logout
- **Route**: `GET /logout`
- **Authentication**: Required
- **On success**: Redirects to `/login`.

---

## Data Model

### ChatSession (SQLite via SQLAlchemy)
| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer (PK) | Auto-increment session ID |
| `user_id` | Integer (FK) | References `users.id` |
| `title` | String (200) | AI-generated title from LLM narrative |
| `scan_filename` | String (300) | Original uploaded filename |
| `scan_data` | Text (JSON) | Full prediction results and narrative |
| `created_at` | DateTime | Session creation timestamp |
| `updated_at` | DateTime | Last activity timestamp |

### ChatMessage (SQLite via SQLAlchemy)
| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer (PK) | Auto-increment message ID |
| `session_id` | Integer (FK) | References `chat_sessions.id` |
| `role` | String (20) | `"user"` or `"assistant"` |
| `content` | Text | Message body |
| `created_at` | DateTime | Message timestamp |