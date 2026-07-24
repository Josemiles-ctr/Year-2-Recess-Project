import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.domain.entities import PredictionResult, ChatMessage
from src.interfaces.gateways import LlmServiceGateway

try:
    import google.generativeai as generativeai
except ImportError:  # pragma: no cover
    generativeai = None


class GeminiLlmService(LlmServiceGateway):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.NullHandler())

        self.provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        self.model = os.getenv("LLM_MODEL", "").strip()
        self.client = None

        self._configure_client()

    def _configure_client(self) -> None:
        if not self.api_key:
            self.logger.warning("GeminiLlmService configured without an LLM API key.")
            return

        if self.provider in ("gemini", "google", "google gemini"):
            if generativeai is None:
                self.logger.warning(
                    "Google Gemini SDK not installed; cannot initialize Gemini client."
                )
                return
            generativeai.configure(api_key=self.api_key)
            self.client = "gemini"
            self.model = self.model or "gemini-1.0"
            return

        if generativeai is not None:
            generativeai.configure(api_key=self.api_key)
            self.client = "gemini"
            self.model = self.model or "gemini-1.0"
            return

        self.logger.warning(
            "GeminiLlmService did not find a supported LLM library; narrative generation will fallback."
        )

    def generate_report_narrative(
        self, traditional_result: PredictionResult, cnn_result: PredictionResult
    ) -> str:
        """Generate a structured clinical narrative for the diagnostic report."""
        context = self._build_diagnostic_context(traditional_result, cnn_result)
        prompt = self._build_prompt(context)

        try:
            text, usage = self._call_llm(prompt)
            self._log_token_usage(usage)
            narrative = self._parse_narrative(text)
            if narrative:
                return narrative
            raise ValueError("LLM response did not return a valid narrative payload.")
        except Exception as exc:  # pragma: no cover
            self.logger.exception("LLM narrative generation failed")
            return self._fallback_narrative(traditional_result, cnn_result, str(exc))

    def _build_diagnostic_context(
        self, traditional_result: PredictionResult, cnn_result: PredictionResult
    ) -> Dict[str, Any]:
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
            "consensus_agreement": traditional_result.prediction_label
            == cnn_result.prediction_label,
            "consensus_verdict": (
                traditional_result.prediction_label
                if traditional_result.prediction_label == cnn_result.prediction_label
                else f"Discordant ({traditional_result.prediction_label} vs {cnn_result.prediction_label})"
            ),
            "average_confidence": round(
                (traditional_result.confidence_score + cnn_result.confidence_score) / 2, 3
            ),
        }

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        system_instruction = (
            "You are a clinical-grade radiology AI assistant writing a cancer screening report for a chest X-ray. "
            "Use a structured, professional tone appropriate for an interdisciplinary clinical audience. "
            "Produce a concise HTML narrative and a structured JSON envelope. "
            "Do not hallucinate findings that are not present in the diagnostic context. "
            "If the content is uncertain, emphasize the need for clinical confirmation and follow-up imaging."
        )

        task_description = (
            "Create an HTML narrative that includes the following sections: "
            "Summary of findings, Detailed explanation of model predictions, Discussion of key features detected, "
            "Clinical recommendations and caveats, and References to detected radiological patterns. "
            "Return output as a JSON object with keys: narrative_html, summary, detailed_explanation, "
            "feature_discussion, recommendations, radiological_references. "
            "The narrative_html value must contain valid HTML markup using headings (for example <h3>) and paragraphs.",
        )

        payload = json.dumps(context, indent=2, ensure_ascii=False)
        return f"{system_instruction}\n\n{task_description}\n\nDiagnostic Context:\n{payload}"

    def _call_llm(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        if self.client == "gemini":
            if generativeai is None:
                raise RuntimeError("Google Gemini SDK is unavailable.")
            response = generativeai.responses.create(
                model=self.model,
                temperature=0.0,
                max_output_tokens=512,
                input=prompt,
            )
            if hasattr(response, "output_text") and response.output_text:
                content = response.output_text.strip()
            else:
                content = str(response)
            usage = {}
            if hasattr(response, "metadata") and isinstance(response.metadata, dict):
                usage = response.metadata.get("tokenUsage", {}) or response.metadata.get(
                    "usage", {}
                )
            return content, usage

        raise RuntimeError("No supported LLM client is configured for GeminiLlmService.")

    def _parse_narrative(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        parsed = self._extract_json_payload(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("narrative_html"), str):
            return parsed["narrative_html"].strip()

        return text.strip()

    def _extract_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return None

    def _log_token_usage(self, usage: Dict[str, Any]) -> None:
        if not usage:
            return
        try:
            prompt_tokens = usage.get("prompt_tokens") or usage.get("promptTokenCount")
            completion_tokens = usage.get("completion_tokens") or usage.get("completionTokenCount")
            total_tokens = usage.get("total_tokens") or usage.get("totalTokenCount")
            self.logger.info(
                "LLM token usage provider=%s model=%s prompt=%s completion=%s total=%s",
                self.client,
                self.model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )
        except Exception:
            self.logger.debug("Unable to parse token usage from LLM response.")

    def _fallback_narrative(
        self,
        traditional_result: PredictionResult,
        cnn_result: PredictionResult,
        error: str,
    ) -> str:
        return (
            "<h3>Diagnostic Narrative Unavailable</h3>"
            "<p>The narrative summary could not be generated by the configured language model.</p>"
            "<p>This report includes the raw model outputs below for review, but it is not a substitute for clinical interpretation.</p>"
            f"<p><strong>Traditional model:</strong> {traditional_result.prediction_label} "
            f"({traditional_result.confidence_score * 100:.1f}% confidence).</p>"
            f"<p><strong>CNN model:</strong> {cnn_result.prediction_label} "
            f"({cnn_result.confidence_score * 100:.1f}% confidence).</p>"
            f"<p><strong>Error:</strong> {error}</p>"
            "<p>Please confirm this scan with a qualified radiologist and consider obtaining follow-up imaging.</p>"
        )

    def chat_follow_up(
        self, history: List[ChatMessage], new_message: str, diagnostic_context: Dict[str, Any]
    ) -> str:
        """Respond to follow-up chat queries using the active diagnostic context."""
        prompt = self._build_chat_prompt(history, new_message, diagnostic_context)

        try:
            text, usage = self._call_llm(prompt)
            self._log_token_usage(usage)
            return text.strip()
        except Exception:  # pragma: no cover
            self.logger.exception("LLM chat follow-up failed")
            return (
                "I am unable to generate a chat response at this time. "
                "Please review the diagnostic context and try again later."
            )

    def _build_chat_prompt(
        self,
        history: List[ChatMessage],
        new_message: str,
        diagnostic_context: Dict[str, Any],
    ) -> str:
        system_instruction = (
            "You are a clinical decision support assistant for radiology. "
            "Use the diagnostic findings and model outputs to answer follow-up questions clearly, "
            "concisely, and with appropriate clinical caution. "
            "If the user asks about confidence, emphasize the model probabilities and the need for expert review. "
            "If the question relates to next steps, recommend clinical validation and additional imaging when appropriate."
        )

        context_payload = json.dumps(diagnostic_context, indent=2, ensure_ascii=False)
        chat_history_text = "\n".join(
            f"{item.role.capitalize()}: {item.content.strip()}" for item in history
        )
        return (
            f"{system_instruction}\n\n"
            f"Diagnostic Context:\n{context_payload}\n\n"
            f"Conversation History:\n{chat_history_text}\n\n"
            f"User Question:\n{new_message.strip()}"
        )
