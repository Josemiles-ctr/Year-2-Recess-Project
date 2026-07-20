from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import datetime


@dataclass
class XRayScan:
    """Domain representation of an uploaded medical scan."""

    filename: str
    image_bytes: bytes
    width: int = 0
    height: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    uploaded_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


@dataclass
class PredictionResult:
    """Domain representation of the outputs of classification engines."""

    prediction_label: str  # e.g., "Malignant" or "Benign"
    confidence_score: float  # Range: [0.0, 1.0]
    extracted_features: Dict[str, float] = field(
        default_factory=dict
    )  # Texture, contrast, LBP parameters
    grad_cam_path: Optional[str] = None  # CNN Activation heatmap location
    model_type: str = "Unknown"  # "Traditional ML" or "Deep CNN"


@dataclass
class DiagnosticReport:
    """Domain representation of a consolidated report with LLM dialogue context."""

    scan_details: XRayScan
    traditional_prediction: PredictionResult
    cnn_prediction: PredictionResult
    consensus_verdict: str
    risk_level: str
    overall_confidence: float
    llm_narrative: str
    diagnosed_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


@dataclass
class ChatMessage:
    """A single turn in the diagnostic assistant chat conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
