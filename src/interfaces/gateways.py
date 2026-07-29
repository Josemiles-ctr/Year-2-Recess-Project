from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from src.domain.entities import XRayScan, PredictionResult, ChatMessage


class TraditionalModelGateway(ABC):
    """Port for Traditional Machine Learning model inference (Features + Classifier)."""

    @abstractmethod
    def extract_features(self, scan: XRayScan) -> Dict[str, float]:
        """Calculates image texture, HOG, LBP, and statistical values."""
        pass

    @abstractmethod
    def predict(self, scan: XRayScan) -> PredictionResult:
        """Extracts features and predicts cancer probability."""
        pass


class CnnModelGateway(ABC):
    """Port for Deep learning model inference (CNN/ResNet)."""

    @abstractmethod
    def predict(self, scan: XRayScan) -> PredictionResult:
        """Feeds image pixels directly into CNN layer matrices and computes prediction."""
        pass

    @abstractmethod
    def generate_grad_cam(self, scan: XRayScan) -> str:
        """Generates gradient-weighted class activation map (Grad-CAM) file path."""
        pass


class LlmServiceGateway(ABC):
    """Port for LLM Diagnostic Dialogues and Report Generation."""

    @abstractmethod
    def generate_report_narrative(
        self, traditional_result: PredictionResult, cnn_result: PredictionResult
    ) -> Tuple[str, str]:
        """Injects model diagnostic metrics to compile natural language report.
        Returns (narrative_html, session_title).
        """
        pass

    @abstractmethod
    def chat_follow_up(
        self, history: List[ChatMessage], new_message: str, diagnostic_context: Dict[str, Any]
    ) -> str:
        """Answers follow-up patient/doctor questions with dynamic diagnostic context."""
        pass

    @abstractmethod
    def validate_chest_xray(self, image_bytes: bytes, filename: str = "") -> Tuple[bool, str]:
        """Verifies using Gemini Vision whether the image is a valid human chest X-ray radiograph.
        Returns (is_chest_xray: bool, reason_or_description: str).
        """
        pass


