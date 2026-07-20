"""Deterministic local simulation services for the AuraScan demonstration flow."""

import hashlib
from typing import Any, Dict, List

from src.domain.entities import ChatMessage, PredictionResult, XRayScan
from src.interfaces.gateways import CnnModelGateway, LlmServiceGateway, TraditionalModelGateway


MOCK_CASES = (
    {
        "traditional": {"label": "Malignant", "confidence": 0.81},
        "cnn": {"label": "Malignant", "confidence": 0.89},
        "features": {
            "mean_intensity": 118.4,
            "std_deviation": 38.6,
            "contrast": 135.0,
            "edge_density": 0.08,
            "spatial_entropy": 3.9,
            "homogeneity": 0.55,
        },
    },
    {
        "traditional": {"label": "Benign", "confidence": 0.22},
        "cnn": {"label": "Benign", "confidence": 0.16},
        "features": {
            "mean_intensity": 96.8,
            "std_deviation": 19.7,
            "contrast": 74.0,
            "edge_density": 0.03,
            "spatial_entropy": 2.8,
            "homogeneity": 0.84,
        },
    },
    {
        "traditional": {"label": "Benign", "confidence": 0.46},
        "cnn": {"label": "Malignant", "confidence": 0.58},
        "features": {
            "mean_intensity": 107.2,
            "std_deviation": 28.9,
            "contrast": 102.0,
            "edge_density": 0.05,
            "spatial_entropy": 3.3,
            "homogeneity": 0.69,
        },
    },
)


def _mock_case(scan: XRayScan) -> Dict[str, Any]:
    """Choose a repeatable fixture without presenting it as image inference."""
    digest = hashlib.sha256(f"{scan.filename}:{len(scan.image_bytes)}".encode()).digest()
    return MOCK_CASES[digest[0] % len(MOCK_CASES)]


class SimulatedTraditionalModel(TraditionalModelGateway):
    """Returns predefined feature-model fixtures for the demonstration flow."""

    def extract_features(self, scan: XRayScan) -> Dict[str, float]:
        return dict(_mock_case(scan)["features"])

    def predict(self, scan: XRayScan) -> PredictionResult:
        case = _mock_case(scan)["traditional"]
        return PredictionResult(
            prediction_label=case["label"],
            confidence_score=case["confidence"],
            extracted_features=self.extract_features(scan),
            model_type="Simulated Feature Model",
        )


class SimulatedCnnModel(CnnModelGateway):
    """Returns predefined CNN fixtures for demos without model weights."""

    def predict(self, scan: XRayScan) -> PredictionResult:
        case = _mock_case(scan)["cnn"]
        return PredictionResult(
            prediction_label=case["label"],
            confidence_score=case["confidence"],
            grad_cam_path=self.generate_grad_cam(scan),
            model_type="Simulated CNN",
        )

    def generate_grad_cam(self, scan: XRayScan) -> str:
        return ""


class SimulatedLlmService(LlmServiceGateway):
    """Creates safe local report and chat text when no external LLM is configured."""

    def generate_report_narrative(
        self, traditional_result: PredictionResult, cnn_result: PredictionResult
    ) -> str:
        return (
            "<h3>Simulation mode</h3>"
            "<p>This demonstration report is generated from a local, deterministic simulation; "
            "it is not a medical diagnosis and must not be used for clinical decisions.</p>"
            f"<p>The feature-model simulation returned <strong>{traditional_result.prediction_label}</strong> "
            f"at {traditional_result.confidence_score * 100:.1f}% confidence. The CNN simulation returned "
            f"<strong>{cnn_result.prediction_label}</strong> at {cnn_result.confidence_score * 100:.1f}% confidence.</p>"
            "<p>Use this screen to demonstrate the application workflow while trained models and clinical validation are added.</p>"
        )

    def chat_follow_up(
        self, history: List[ChatMessage], new_message: str, diagnostic_context: Dict[str, Any]
    ) -> str:
        verdict = diagnostic_context.get("verdict", "Unavailable")
        risk = diagnostic_context.get("risk_level", "Unavailable")
        question = new_message.lower()
        if "confidence" in question or "accur" in question:
            detail = (
                f"The simulated feature model is {diagnostic_context.get('traditional_confidence', 0) * 100:.1f}% "
                f"and the simulated CNN is {diagnostic_context.get('cnn_confidence', 0) * 100:.1f}% confident."
            )
        elif "risk" in question or "result" in question:
            detail = f"The demonstration consensus is {verdict} with a {risk} simulated risk level."
        else:
            detail = f"The active demonstration result is {verdict} with a {risk} simulated risk level."
        return (
            f"{detail} This is a simulated application response, not medical advice. "
            "Please consult a qualified clinician for interpretation of an actual scan."
        )
