import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from src.infrastructure.web.routes import web_bp
from src.infrastructure.llm.gemini_service import GeminiRagService
from src.infrastructure.session_store import SessionStore


def create_app() -> Flask:
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")

    app = Flask(
        "src",
        template_folder=str(project_root / "src" / "templates"),
        static_folder=str(project_root / "src" / "static"),
    )
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "development-only-secret-key")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    rag_service = GeminiRagService()
    app.config["RAG_SERVICE"] = rag_service

    session_store = SessionStore()
    app.config["SESSION_STORE"] = session_store

    app.register_blueprint(web_bp)

    return app
