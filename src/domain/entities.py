from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import datetime


class InvalidImageError(ValueError):
    """Raised when an uploaded file is empty, corrupt, or not a valid image format."""
    pass


class NotAnXRayError(ValueError):
    """Raised when an uploaded image is valid, but is not a recognized human chest radiograph (X-ray)."""
    pass


def _utc_now() -> datetime.datetime:
    """Helper function to return current timezone-aware UTC timestamp."""
    return datetime.datetime.now(datetime.timezone.utc)


@dataclass
class XRayScan:
    """Domain representation of an uploaded medical scan."""

    filename: str
    image_bytes: bytes
    width: int = 0
    height: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    uploaded_at: datetime.datetime = field(default_factory=_utc_now)


@dataclass
class PredictionResult:
    """Domain representation of the outputs of classification engines."""

    prediction_label: str  # e.g., "Malignant" or "Benign", or top pathology
    confidence_score: float  # Range: [0.0, 1.0]
    extracted_features: Dict[str, float] = field(
        default_factory=dict
    )  # Texture, contrast, LBP parameters
    grad_cam_path: Optional[str] = None  # CNN Activation heatmap location
    model_type: str = "Unknown"  # "Traditional ML" or "Deep CNN"
    per_class_probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Domain representation of a consolidated report with LLM dialogue context."""

    scan_details: XRayScan
    traditional_prediction: PredictionResult
    cnn_prediction: PredictionResult
    llm_narrative: str
    session_title: str = ""
    diagnosed_at: datetime.datetime = field(default_factory=_utc_now)


@dataclass
class ChatMessage:
    """A single turn in the diagnostic assistant chat conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime.datetime = field(default_factory=_utc_now)
