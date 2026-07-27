# AuraScan — AI-Powered Chest X-Ray Diagnostics
### BSE2301: Software Engineering Mini Project 2 — Recess 2026 (Group O)

A clinical decision-support workspace for chest X-ray diagnostics using dual ML models (Random Forest + CNN) and an LLM-powered diagnostic assistant. Built with Flask, SQLite, and Google Gemini.

---

## Table of Contents
1. Project Overview
2. Key Features
3. System Architecture
4. Repository Structure
5. Auth & Navigation Flow
6. Setup & Installation
7. Running the Application
8. Group Members

---

## Project Overview

AuraScan brings traditional feature analysis and neural-network outputs into one structured review, with a clear report and follow-up assistant. Users upload a chest X-ray, receive dual-model predictions (Random Forest + CNN) side by side, read an AI-generated clinical narrative, and can ask follow-up questions to the diagnostic assistant.

---

## Key Features

- **Dual-Model Predictions**: Random Forest (feature-based) and CNN predictions displayed side-by-side
- **15-Class Pathology Classification**: NIH ChestX-ray14 categories (Atelectasis, Consolidation, Infiltration, Mass, Nodule, etc.)
- **AI Clinical Narrative**: LLM-generated structured report in HTML
- **Interactive Chat Assistant**: Context-aware follow-up Q&A with session memory
- **Session Management**: Persistent scan history with created date tracking
- **Responsive Dashboard**: Sidebar navigation, session list, and auto-redirect to latest report
- **Secure Auth**: Email/password registration with Flask-Login session management

---

## Auth & Navigation Flow

```
Index (/)
  ├── Not authenticated → Show landing page → "Get Started" → Login → Report
  ├── Authenticated + has sessions → Redirect to Report Dashboard (/report?sid=<latest>)
  └── Authenticated + no sessions → Redirect to Upload Page (/upload)

Login/Register → Redirect to Report Dashboard
Report Dashboard
  ├── Has sessions → Show predictions, AI summary, chat
  └── No sessions → Redirect to Upload Page

Upload Page
  ├── Has previous sessions → Show "View Reports" button
  └── New user → Upload first scan → Auto-redirect to Report
```

---

## System Architecture

Clean Architecture (layered):

```
src/
├── domain/          — Core entities (XRayScan, PredictionResult, ChatMessage)
├── use_cases/       — Business logic (PredictCancerUseCase, ChatWithAssistantUseCase)
├── interfaces/      — Adapters (Controllers, Gateways/ABCs)
└── infrastructure/  — Flask, SQLAlchemy, Gemini API, ML models
```

Key design patterns: Dependency Injection, Gateway/Repository, ABC boundaries.

---

## Repository Structure

```
├── assets/                    # Background images, metadata CSVs
├── docs/                      # API spec, architecture docs
│   ├── api_spec.md
│   ├── architecture.md
│   └── report_outline.md
├── models/                    # Trained ML models (.keras, .joblib)
├── notebooks/                 # Jupyter notebooks (EDA, model training)
├── tests/                     # Test suite
├── api/index.py               # Vercel serverless entry point
└── src/                       # Main application package
    ├── app.py                 # Flask app entry point
    ├── domain/
    │   └── entities.py        # XRayScan, PredictionResult, ChatMessage
    ├── use_cases/
    │   ├── predict.py         # Dual-model prediction orchestration
    │   └── chat.py            # LLM chat orchestration
    ├── interfaces/
    │   ├── controllers.py     # AnalyzeController, ChatController
    │   └── gateways.py        # Abstract gateways (TraditionalModel, CNN, LLM)
    ├── infrastructure/
    │   ├── database.py        # SQLAlchemy models (User, ChatSession, ChatMessage)
    │   ├── ml/                # Concrete ML model wrappers
    │   │   ├── keras_cnn_model.py
    │   │   ├── sklearn_model.py
    │   │   └── pytorch_model.py
    │   ├── llm/
    │   │   └── gemini_service.py  # Google Gemini integration
    │   └── web/
    │       ├── __init__.py
    │       ├── app_setup.py   # Flask app factory, LoginManager
    │       └── routes.py      # All HTTP routes (auth, pages, API)
    ├── templates/             # Jinja2 HTML templates
    │   ├── index.html         # Landing page
    │   ├── login.html         # Sign In
    │   ├── register.html      # Create Account
    │   ├── upload.html        # Scan upload
    │   └── report.html        # Report dashboard + chat
    └── static/
        ├── css/main.css       # All application styles
        ├── images/            # Logo, backgrounds
        └── js/chat.js         # Client-side upload, chat, sidebar
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
git clone <repository-url>
cd Year-2-Recess
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the root directory:
```env
SECRET_KEY=your_secret_key_here
LLM_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Running the Application

```bash
python src/app.py
# Or
flask --app src.app run
```

Open http://127.0.0.1:5000 in your browser.

---

## Security

- Credentials stored in `.env` (excluded from version control)
- GitHub Actions runs **TruffleHog** on every push to detect exposed secrets
- Python code formatted and linted with **Ruff** (`ruff format src`, `ruff check src`)

---

## Group Members

| Name | Registration Number |
| --- | --- |
| OTAI JOSEPH | 24/U/23001 |
| AKATUKUNDA PRECIOUS PRAISE | 24/U/0147 |
| ABUREK EMMANUEL | 24/U/02614/PS |
| AGABA DORECK | 24/U/23685/PS |
| Kibenge Victor Bulasio | 24/U/0544 |

For inquiries: jeff.geoff.mis@gmail.com  
GitHub: https://github.com/Josemiles-ctr/Year-2-Recess-Project
