import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from src.infrastructure.web.routes import web_bp
from src.infrastructure.simulation import (
    SimulatedCnnModel,
    SimulatedLlmService,
    SimulatedTraditionalModel,
)
from src.interfaces.controllers import AnalyzeController, ChatController
from src.use_cases.chat import ChatWithAssistantUseCase
from src.use_cases.predict import PredictCancerUseCase


def create_app() -> Flask:
    """Application factory that wires together Clean Architecture dependencies and starts Flask.

    Assignee Guidelines:
    1. Read environmental settings (.env configurations).
    2. Initialize the Flask application container.
    3. Instantiate the concrete gateways (Traditional Model, PyTorch CNN, Gemini LLM Service).
    4. Instantiate use cases (Predict Cancer, Chat with Assistant) and inject the gateways.
    5. Instantiate controller adapters (Analyze Controller, Chat Controller) and inject use cases.
    6. Attach controller adapters to Flask application config context.
    7. Register routes blueprints.
    """
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")

    app = Flask(
        "src",
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "development-only-secret-key")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    traditional_model = SimulatedTraditionalModel()
    cnn_model = SimulatedCnnModel()
    llm_service = SimulatedLlmService()
    app.config["ANALYZE_CONTROLLER"] = AnalyzeController(
        PredictCancerUseCase(traditional_model, cnn_model, llm_service)
    )
    app.config["CHAT_CONTROLLER"] = ChatController(ChatWithAssistantUseCase(llm_service))
    app.register_blueprint(web_bp)

    return app
