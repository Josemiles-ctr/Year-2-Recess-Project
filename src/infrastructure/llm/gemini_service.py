import logging
import os
import re
from typing import Any, Dict, List

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from src.domain.entities import ChatMessage, PredictionResult
from src.interfaces.gateways import LlmServiceGateway


CHAT_MODEL = "gemini-flash-latest"


class GeminiLlmService(LlmServiceGateway):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.NullHandler())

        api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)
        self.model = (os.getenv("LLM_MODEL") or CHAT_MODEL).strip()

    def generate_report_narrative(
        self, traditional_result: PredictionResult, cnn_result: PredictionResult
    ) -> tuple[str, str]:
        context = self._build_diagnostic_context(traditional_result, cnn_result)
        prompt = self._build_prompt(context)
        try:
            text = self._call_llm(prompt)
            narrative, title = self._parse_narrative(text)
            return narrative or text, title or self._default_title(context)
        except Exception as exc:
            self.logger.exception("LLM narrative generation failed")
            return self._fallback_narrative(traditional_result, cnn_result, str(exc)), ""

    @staticmethod
    def _default_title(context: Dict[str, Any]) -> str:
        """Generate a concise plain-text title using feature and visual analysis findings."""
        trad = context.get("traditional_prediction", "")
        cnn = context.get("cnn_prediction", "")
        if trad == cnn:
            return f"{trad} scan detected"
        return f"{trad} / {cnn}"

    def _build_diagnostic_context(
        self, traditional_result: PredictionResult, cnn_result: PredictionResult
    ) -> Dict[str, Any]:
        """Construct structured diagnostic payload for prompt construction."""
        features = traditional_result.extracted_features or {}
        return {
            "traditional_model_type": traditional_result.model_type,
            "traditional_prediction": traditional_result.prediction_label,
            "traditional_confidence": round(traditional_result.confidence_score, 3),
            "traditional_features": {
                key: round(value, 4) if isinstance(value, (int, float)) else value
                for key, value in features.items()
            },
            "cnn_model_type": cnn_result.model_type,
            "cnn_prediction": cnn_result.prediction_label,
            "cnn_confidence": round(cnn_result.confidence_score, 3),
            "cnn_grad_cam_path": cnn_result.grad_cam_path or "Unavailable",
            "cnn_per_class_probabilities": cnn_result.per_class_probabilities,
            "consensus_agreement": traditional_result.prediction_label
            == cnn_result.prediction_label,
        }

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt instructing LLM to write a purely biological & anatomical report with ZERO technical jargon."""
        import json

        system = (
            "You are an expert clinical radiologist writing a diagnostic evaluation for a chest radiograph. "
            "Your report narrative MUST be strictly biological, anatomical, and physiological. "
            "Describe all findings using thoracic anatomical structures (lungs, parenchymal fields, hilar structures, pleura, mediastinum, diaphragm, cardiac silhouette, thoracic cavity) "
            "and biological/pathological processes (tissue opacity, consolidation, infiltration, effusion, nodular densities, atelectasis, etc.). "
            "STRICT PROHIBITION: Do NOT use any technical computer science, software, or machine-learning terms (such as model, prediction, confidence, probability, algorithm, machine learning, AI, feature, data point, extraction, CNN, Random Forest, percentage score, etc.). "
            "Frame all findings as direct anatomical observations and physiological interpretations. "
            "Produce a concise HTML report with the following biological section headings: "
            "<h3>Anatomical & Diagnostic Overview</h3>, "
            "<h3>Detailed Biological & Tissue Observations</h3>, "
            "<h3>Clinical Impression & Recommendations</h3>. "
            "Return the response as valid JSON with two keys: "
            '"title" (a short, plain-text biological/clinical diagnosis label for this scan, at most 7 words, no HTML, not starting with "Summary"), '
            'and "narrative_html" (the full report as valid HTML using h3/h4 headings and paragraphs). '
            "Do not hallucinate findings not supported by the clinical context."
        )
        return (
            f"{system}\n\nDiagnostic Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}"
        )

    def _call_llm(self, prompt: str) -> str:
        try:
            resp = self.client.models.generate_content(model=self.model, contents=prompt)
            return resp.text.strip() if resp.text else ""
        except genai_errors.ClientError as e:
            raise RuntimeError(self._friendly_error(e)) from e

    def _parse_narrative(self, text: str) -> tuple[str, str]:
        if not text or not text.strip():
            return "", ""
        import json as _json

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                payload = _json.loads(candidate)
                narrative = (payload.get("narrative_html") or payload.get("narrative", "")).strip()
                title = (payload.get("title") or "").strip()
                if narrative:
                    return narrative, title
            except _json.JSONDecodeError:
                pass
        return text.strip(), ""

    def _fallback_narrative(
        self, traditional_result: PredictionResult, cnn_result: PredictionResult, error: str
    ) -> str:
        """Construct fallback biological narrative when report generation service is unavailable."""
        return (
            "<h3>Anatomical Report Unavailable</h3>"
            "<p>The biological narrative summary could not be generated at this time.</p>"
            f"<p><strong>Primary Anatomical Observation:</strong> {traditional_result.prediction_label}.</p>"
            f"<p><strong>Secondary Visual Observation:</strong> {cnn_result.prediction_label}.</p>"
            "<p>Please consult a qualified radiologist for a thorough clinical examination.</p>"
        )

    def chat_follow_up(
        self, history: List[ChatMessage], new_message: str, diagnostic_context: Dict[str, Any]
    ) -> str:
        prompt = self._build_chat_prompt(history, new_message, diagnostic_context)
        try:
            return self._call_llm(prompt)
        except Exception:
            self.logger.exception("LLM chat follow-up failed")
            return (
                "I am unable to generate a chat response at this time. "
                "Please review the diagnostic context and try again later."
            )

    def _build_chat_prompt(
        self, history: List[ChatMessage], new_message: str, diagnostic_context: Dict[str, Any]
    ) -> str:
        """Construct system prompt for chat follow-up queries strictly using biological, anatomical, and clinical radiologic terms."""
        import json

        system = (
            "You are an expert clinical radiology consultant answering questions about a chest radiograph. "
            "Respond purely using biological, anatomical, physiological, and medical radiologic terms. "
            "Discuss findings in relation to thoracic anatomy (lungs, hilar regions, pleura, mediastinum, tissue parenchyma, cardiac silhouette) and disease biology. "
            "STRICT PROHIBITION: Do NOT use any technical computer science, software, or AI/machine-learning terms (such as model, prediction, confidence, probability score, percentage, algorithm, CNN, Random Forest, feature extraction, AI, machine learning, data point, etc.). "
            "Explain all observations as natural biological and anatomical characteristics of the patient's scan. "
            "Keep responses helpful, professional, and grounded in clinical medicine. "
            "You remember the full conversation history - build on previous exchanges naturally. "
            "If the user asks something not covered by the context, say so clearly. "
            "Always note that findings require clinical confirmation by an attending physician or radiologist."
        )
        context_payload = json.dumps(diagnostic_context, indent=2, ensure_ascii=False)
        chat_history_text = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content.strip()}" for m in history
        )
        return (
            f"{system}\n\n"
            f"Diagnostic Context:\n{context_payload}\n\n"
            f"Conversation History:\n{chat_history_text}\n\n"
            f"User Question:\n{new_message.strip()}"
        )

    @staticmethod
    def _friendly_error(err: genai_errors.ClientError) -> str:
        msg = str(err)
        code = getattr(err, "code", 0) or 0
        if not code:
            match = re.search(r"^(\d+)", msg)
            code = int(match.group(1)) if match else 0
        if code == 429:
            return "The AI service is temporarily out of requests due to quota limits. Please wait and try again."
        if code == 403:
            return "The AI service couldn't authenticate your API key. Please check your configuration."
        if code == 400:
            return "The request to the AI service was invalid. Try rephrasing your question."
        if code == 404:
            return "The AI model is not available. The service may be updating."
        return "The AI service returned an unexpected error. Please try again later."

    def validate_chest_xray(self, image_bytes: bytes, filename: str = "") -> tuple[bool, str]:
        """Validate whether an image is a human chest radiograph using Gemini Vision multimodal capabilities.

        Args:
            image_bytes: Raw binary content of the uploaded image file.
            filename: Optional filename used for context logging.

        Returns:
            Tuple of (is_chest_xray: bool, reason: str).
        """
        if not image_bytes:
            return False, "The uploaded file is empty."

        import io
        import json
        from PIL import Image

        # 1. Determine image MIME type via Pillow header inspection
        mime_type = "image/jpeg"
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                fmt = (img.format or "JPEG").lower()
                if fmt == "jpg":
                    fmt = "jpeg"
                mime_type = f"image/{fmt}"
        except Exception:
            mime_type = "image/jpeg"

        # 2. Prepare structured system prompt for vision-based anatomical assessment
        prompt = (
            "You are an expert radiology AI system evaluating image suitability for chest X-ray screening.\n"
            "Analyze the provided image and determine if it is a human chest radiograph (Chest X-Ray scan).\n"
            "Confirm the presence of thoracic anatomy (lungs, rib cage, heart silhouette, spine, or clavicles).\n"
            "Reject non-medical images (e.g. pets, scenery, graphics, documents, text) and non-chest scans (e.g. brain MRI, extremity X-ray, abdominal ultrasound).\n\n"
            "Return strictly valid JSON with the following two keys:\n"
            '{\n  "is_chest_xray": boolean,\n  "reason": "1-2 sentence explanation of why it is or is not a chest X-ray"\n}'
        )

        try:
            # 3. Construct multimodal input part and call Gemini models API
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[image_part, prompt],
            )
            text = resp.text.strip() if resp.text else ""

            # 4. Parse structured JSON output from Gemini response
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                payload = json.loads(text[start : end + 1])
                is_xray = bool(payload.get("is_chest_xray", False))
                reason = str(payload.get("reason", "Validation check completed."))
                return is_xray, reason

            return False, "Could not parse verification response from vision service."
        except Exception as exc:
            self.logger.exception("Gemini Vision validation call failed for %s", filename)
            # Log error and return false with exception context
            return (
                False,
                f"Vision verification service error: {self._friendly_error(exc) if isinstance(exc, genai_errors.ClientError) else str(exc)}",
            )
