import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
from flask_login import LoginManager

from src.infrastructure.database import db, User
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

    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{project_root / 'aurascan.db'}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "web.login"

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    with app.app_context():
        db.create_all()

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
