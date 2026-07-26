import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
from flask_session import Session

from src.infrastructure.web.routes import web_bp
from src.infrastructure.ml.keras_cnn_model import KerasCnnModel
from src.infrastructure.ml.sklearn_model import SklearnTraditionalModel
from src.infrastructure.llm.gemini_service import GeminiLlmService
from src.interfaces.controllers import AnalyzeController, ChatController
from src.use_cases.chat import ChatWithAssistantUseCase
from src.use_cases.predict import PredictCancerUseCase


def create_app() -> Flask:
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")

    app = Flask(
        "src",
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "development-only-secret-key")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = str(project_root / "flask_session")
    app.config["SESSION_PERMANENT"] = True
    Session(app)

    model_path = str(project_root / "models" / "nih_chest_xray_cnn_model.keras")
    cnn_model = KerasCnnModel(model_path=model_path)
    traditional_model = SklearnTraditionalModel()
    llm_service = GeminiLlmService()

    app.config["ANALYZE_CONTROLLER"] = AnalyzeController(
        PredictCancerUseCase(traditional_model, cnn_model, llm_service)
    )
    app.config["CHAT_CONTROLLER"] = ChatController(ChatWithAssistantUseCase(llm_service))
    app.register_blueprint(web_bp)

    return app
