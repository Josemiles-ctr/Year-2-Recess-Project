# AuraScan — Clinical Web Interface & AI Diagnostic Integration
## BSE2301 Software Engineering Mini Project 2 — Group O

---

## 1. Introduction

AuraScan is a clinical decision-support web application that brings dual-model AI diagnostics for chest X-ray analysis into a single, structured workspace. Traditional medical imaging tools return a binary label or a single probability score. AuraScan goes further by comparing predictions from two fundamentally different approaches — a feature-engineered Random Forest classifier and a deep convolutional neural network (CNN) — and presenting the results side by side for clinical review.

The application is built using **Flask** (Python) following **Clean Architecture** principles, with **SQLite** for persistent session storage and **Google Gemini** as the LLM backend for generating diagnostic narratives and powering an interactive chat assistant. The frontend uses server-rendered **Jinja2** templates with a custom dark-theme CSS design system, **Lucide** icons, and vanilla JavaScript for uploads, chat, and session management.

The system addresses a key gap in automated radiology workflows: providing clinicians with both the statistical rigor of traditional ML features (texture, HOG, LBP) and the pattern-recognition power of deep learning, unified with a natural language interface for follow-up questions.

### Key Objectives

- Provide a dual-model diagnostic pipeline (Random Forest + CNN) with side-by-side comparison
- Generate structured, clinician-readable AI narratives from model outputs
- Enable context-aware follow-up Q&A through an LLM-powered chat assistant
- Maintain persistent session history with secure user authentication
- Follow clean architecture to keep business logic independent of frameworks

---

## 2. System Architecture

AuraScan follows **Uncle Bob's Clean Architecture**, organizing code into concentric layers where dependencies point inward:

```
┌─────────────────────────────────────────┐
│  Infrastructure (Flask, SQLite, Gemini)  │
│  ┌───────────────────────────────────┐  │
│  │  Interface Adapters (Controllers)  │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │  Use Cases (Business Logic)  │  │  │
│  │  │  ┌───────────────────────┐  │  │  │
│  │  │  │  Domain (Entities)     │  │  │  │
│  │  │  └───────────────────────┘  │  │  │
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Layer Breakdown

**Domain Layer** (`src/domain/entities.py`):
Core business entities with no external dependencies:
- `XRayScan` — Holds binary image data and metadata
- `PredictionResult` — Encapsulates model outputs (label, confidence, per-class probabilities)
- `ChatMessage` — Represents a single conversational turn

**Use Cases Layer** (`src/use_cases/`):
Application-specific business rules:
- `PredictCancerUseCase` — Coordinates dual-model prediction and LLM narrative generation
- `ChatWithAssistantUseCase` — Manages conversational follow-up with diagnostic context

**Interface Adapters Layer** (`src/interfaces/`):
Translators between use cases and external systems:
- `AnalyzeController` — Adapts HTTP uploads to prediction use case
- `ChatController` — Adapts chat requests to LLM use case
- `TraditionalModelGateway` (ABC) — Port for feature-based ML
- `CnnModelGateway` (ABC) — Port for deep learning inference
- `LlmServiceGateway` (ABC) — Port for LLM interactions

**Infrastructure Layer** (`src/infrastructure/`):
Concrete implementations of all gateways and framework code:
- `Flask Web Server` — HTTP routing, authentication, session management
- `SQLAlchemy` — Database models (`User`, `ChatSession`, `ChatMessage`)
- `SklearnModel` / `KerasCnnModel` — ML inference wrappers
- `GeminiLlmService` — Google Gemini API integration

### Data Flow

```
User Uploads X-Ray
       │
       ▼
Flask Route (/api/analyze)
       │
       ▼
AnalyzeController.handle_upload()
       │
       ▼
PredictCancerUseCase.execute()
       ├── TraditionalModelGateway.predict() → RF prediction
       ├── CnnModelGateway.predict() → CNN prediction
       └── LlmServiceGateway.generate_report_narrative() → AI summary
       │
       ▼
ChatSession created in SQLite (title, scan_data, filename)
       │
       ▼
Redirect to /report?sid=<id>
       │
       ▼
Report dashboard: RF | CNN (side by side) → AI Summary → Chat Assistant
```

---

## 3. Authentication & User Flow

AuraScan uses **Flask-Login** for session-based authentication with Werkzeug password hashing (PBKDF2).

### User Flow

```
Landing Page (/)
    │
    ├── Not authenticated → Show landing page
    │       └── "Get Started" → Login page → Authenticate → Report Dashboard
    │
    ├── Authenticated + has sessions → Redirect to Report Dashboard
    │       └── Shows latest scan report with predictions, summary, and chat
    │
    └── Authenticated + no sessions → Redirect to Upload Page
            └── Upload first scan → Auto-redirect to Report Dashboard
```

### Routes

| Route | Method | Auth | Description |
|-------|--------|------|-------------|
| `/` | GET | No | Landing page / smart redirect |
| `/login` | GET/POST | No | Email + password login |
| `/register` | GET/POST | No | Account creation |
| `/logout` | GET | Yes | Session termination |
| `/upload` | GET | Yes | X-ray upload page |
| `/report` | GET | Yes | Diagnostic dashboard + chat |

### API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/analyze` | POST | Yes | Upload & analyze X-ray |
| `/api/chat` | POST | Yes | Send chat message |
| `/api/chat/clear` | POST | Yes | Clear chat history |
| `/api/sessions` | GET | Yes | List all sessions |
| `/api/sessions/<id>` | GET | Yes | Get session detail |
| `/api/sessions/<id>` | DELETE | Yes | Delete session |
| `/api/sessions/<id>/title` | PUT | Yes | Update session title |

---

## 4. Key Features

### 4.1 Dual-Model Side-by-Side Comparison

The report dashboard displays Random Forest and CNN predictions in a shared row, allowing direct visual comparison of:
- Top prediction label and confidence score (with progress bar)
- Per-class pathology probabilities across all 15 NIH ChestX-ray categories
- Color-coded confidence levels (high/medium/low)

### 4.2 AI-Generated Clinical Narrative

A structured HTML narrative is generated by Google Gemini, summarizing findings, discussing key features detected, and providing clinical recommendations. This spans the full width below the model comparison row.

### 4.3 Interactive Diagnostic Assistant

A persistent chat panel allows users to ask follow-up questions about the scan. The assistant has access to:
- Full diagnostic context (both model predictions and per-class probabilities)
- Complete conversation history (multi-turn memory)
- The AI summary narrative

### 4.4 Session Management

All scans are persisted in SQLite with:
- AI-generated title (first line of LLM narrative, truncated)
- Original filename and creation timestamp
- Full prediction data and chat history
- Sidebar navigation with session list sorted by recency

### 4.5 Responsive Design

Dark-theme clinical UI with glassmorphism effects, built with:
- CSS custom properties (design tokens)
- Lucide icon system
- Mobile-responsive sidebar with overlay
- Progress indicators for upload and analysis

---

## 5. Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend Framework | Flask 3.1 (Python) |
| Database | SQLite via Flask-SQLAlchemy |
| Authentication | Flask-Login + Werkzeug |
| ML — Traditional | Scikit-learn (Random Forest, feature extraction) |
| ML — Deep Learning | TensorFlow/Keras (CNN, Grad-CAM) |
| LLM | Google Gemini (gemini-flash-latest) |
| Templating | Jinja2 |
| Icons | Lucide 0.468 |
| Fonts | Google Fonts (Inter + Outfit) |
| Styling | Custom CSS (dark theme, glassmorphism) |
| Client-side | Vanilla JavaScript |
| Code Quality | Ruff (formatting + linting) |
| CI/CD | GitHub Actions (TruffleHog, Ruff checks) |

---

## 6. Database Schema

### User
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| email | String (120) | Unique, indexed |
| password_hash | String (256) | Werkzeug hash |
| created_at | DateTime | Registration timestamp |

### ChatSession
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| user_id | Integer (FK) | References users |
| title | String (200) | AI-generated title |
| scan_filename | String (300) | Original filename |
| scan_data | Text (JSON) | Predictions + narrative |
| created_at | DateTime | Session created |
| updated_at | DateTime | Last activity |

### ChatMessage
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| session_id | Integer (FK) | References sessions |
| role | String (20) | "user" or "assistant" |
| content | Text | Message body |
| created_at | DateTime | Message timestamp |

---

## 7. Project Structure

```
src/
├── app.py                        # Flask entry point
├── domain/
│   └── entities.py               # XRayScan, PredictionResult, ChatMessage
├── use_cases/
│   ├── predict.py                # PredictCancerUseCase
│   └── chat.py                   # ChatWithAssistantUseCase
├── interfaces/
│   ├── controllers.py            # AnalyzeController, ChatController
│   └── gateways.py               # Abstract gateways
├── infrastructure/
│   ├── database.py               # SQLAlchemy models
│   ├── ml/
│   │   ├── keras_cnn_model.py    # CNN inference
│   │   ├── sklearn_model.py      # RF inference
│   │   └── pytorch_model.py      # Alternative backend
│   ├── llm/
│   │   └── gemini_service.py     # Gemini integration
│   └── web/
│       ├── app_setup.py          # Flask factory, LoginManager
│       └── routes.py             # All routes and API endpoints
├── templates/
│   ├── index.html                # Landing page
│   ├── login.html                # Sign in
│   ├── register.html             # Create account
│   ├── upload.html               # Scan upload
│   └── report.html               # Report dashboard + chat
└── static/
    ├── css/main.css              # Complete style system
    ├── images/                   # Logo and backgrounds
    └── js/chat.js                # Upload, chat, sidebar logic
```

---

## 8. Setup Instructions

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
git clone https://github.com/Josemiles-ctr/Year-2-Recess-Project
cd Year-2-Recess
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root:
```env
SECRET_KEY=your_secret_key
GEMINI_API_KEY=your_gemini_api_key
```

### Run
```bash
python src/app.py
```
Open http://127.0.0.1:5000

---

## 9. Screenshots

<!-- Screenshots to be inserted by student -->
<!-- Suggested screenshots: -->
<!-- 1. Landing page hero section -->
<!-- 2. Upload page with dropzone -->
<!-- 3. Upload progress bar active -->
<!-- 4. Report dashboard: RF + CNN side by side -->
<!-- 5. AI Summary section -->
<!-- 6. Chat interaction with diagnostic assistant -->
<!-- 7. Login/Register pages -->
<!-- 8. Mobile responsive view -->

---

## 10. Source Code

The complete source code is available on GitHub:
**https://github.com/Josemiles-ctr/Year-2-Recess-Project**

Key files:
- `src/infrastructure/web/routes.py` — All HTTP routes and API endpoints
- `src/use_cases/predict.py` — Dual-model prediction orchestration
- `src/use_cases/chat.py` — LLM chat orchestration
- `src/infrastructure/llm/gemini_service.py` — Gemini prompt engineering
- `src/templates/report.html` — Report dashboard layout
- `src/static/js/chat.js` — Client-side interactions
- `src/static/css/main.css` — Complete design system

---

## 11. Group Members

| Name | Registration Number |
|------|-------------------|
| OTAI JOSEPH | 24/U/23001 |
| AKATUKUNDA PRECIOUS PRAISE | 24/U/0147 |
| ABUREK EMMANUEL | 24/U/02614/PS |
| AGABA DORECK | 24/U/23685/PS |

**Supervisor Contact:** jeff.geoff.mis@gmail.com