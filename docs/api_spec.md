# Web API Specification & Documentation

This document describes the endpoints, input parameters, session payloads, and response JSON formats for the Flask web service.

---

## Web Interface Routes

### 1. Main Dashboard
- **Route**: `GET /`
- **Description**: Returns the main dashboard interface. Contains the file upload dropzone, model comparison displays, and placeholder templates for results.
- **Response**: HTML template (index.html).

### 2. Diagnosis Report View
- **Route**: `GET /report`
- **Description**: Serves a detailed, print-friendly, and interactive summary page containing specific diagnostic logs, metrics tables, and the context-aware chat interface.
- **Response**: HTML template (report.html).

---

## API Endpoints

### 1. Analyze X-Ray Scan
Processes the uploaded image, triggers the dual-model classification engines, and responds with prediction scores and image attributes.

- **Route**: `POST /api/analyze`
- **Content-Type**: `multipart/form-data`
- **Request Parameters**:
  | Key | Type | Required | Description |
  | :--- | :--- | :--- | :--- |
  | `file` | Binary | Yes | The chest/body X-ray image (supported formats: PNG, JPG, JPEG). |

- **Example Response (200 OK)**:
  ```json
  {
    "status": "success",
    "filename": "patient_xray_99182.png",
    "timestamp": "2026-07-20T12:50:00Z",
    "traditional_model": {
      "prediction": "Malignant",
      "confidence": 0.824,
      "features": {
        "contrast": 18.23,
        "homogeneity": 0.35,
        "energy": 0.08,
        "correlation": 0.74,
        "hog_mean": 0.19,
        "lbp_mean": 4.12
      }
    },
    "cnn_model": {
      "prediction": "Malignant",
      "confidence": 0.895,
      "grad_cam_url": "/static/temp/gradcam_99182.png"
    },
    "consensus": {
      "verdict": "Malignant",
      "risk_level": "High",
      "confidence": 0.895
    }
  }
  ```

- **Error Response (400 Bad Request)**:
  ```json
  {
    "status": "error",
    "message": "No file uploaded or file format not supported."
  }
  ```

---

## Chat with AI Assistant
Leverages the predictive outputs stored in the Flask session to answer context-aware follow-up questions from the user or clinician.

- **Route**: `POST /api/chat`
- **Content-Type**: `application/json`
- **Request Headers**:
  - `Content-Type: application/json`
- **Request Payload**:
  ```json
  {
    "message": "Why did the traditional classifier flag this image as malignant?"
  }
  ```

- **Session Context Enclosed (Handled internally)**:
  ```json
  {
    "context": {
      "traditional_model_prediction": "Malignant",
      "traditional_confidence": 0.824,
      "cnn_model_prediction": "Malignant",
      "cnn_confidence": 0.895,
      "homogeneity": 0.35,
      "contrast": 18.23
    }
  }
  ```

- **Example Response (200 OK)**:
  ```json
  {
    "status": "success",
    "response": "The traditional classifier labeled this scan as malignant primarily due to high image contrast (18.23) and low homogeneity (0.35) inside the tissue matrix. This combination often points to highly irregular pixel regions, which are common in cancerous mass densities. The CNN model supports this classification with a higher confidence of 89.5%."
  }
  ```

- **Error Response (500 Internal Server Error)**:
  ```json
  {
    "status": "error",
    "message": "LLM service unavailable. Check API keys."
  }
  ```

---

## Session Management
To facilitate multi-turn chat without requiring a database, the system stores diagnosis records in Flask Client-side Sessions (cryptographically signed cookie).
- **Session Keys**:
  - `active_prediction`: Stores the stringified JSON details of the latest `/api/analyze` call.
  - `chat_history`: Array of chat dicts: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`.
- **Session Clearing**: Session details are automatically cleared when a new image upload is successfully processed, starting a new patient assessment context.
