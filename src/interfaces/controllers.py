from typing import Dict, Any, List
from src.domain.entities import ChatMessage
from src.use_cases.predict import PredictCancerUseCase
from src.use_cases.chat import ChatWithAssistantUseCase


class AnalyzeController:
    """Controller to adapt file uploads into cancer prediction use cases.

    Assignee Guidelines:
    1. Parse files and inputs from HTTP requests.
    2. Invoke prediction use cases.
    3. Transform domain output records into JSON responses.
    """

    def __init__(self, predict_use_case: PredictCancerUseCase):
        self.predict_use_case = predict_use_case

    def handle_upload(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        report = self.predict_use_case.execute(filename, file_bytes)
        cnn_probabilities = report.cnn_prediction.per_class_probabilities
        rf_probabilities = report.traditional_prediction.per_class_probabilities
        return {
            "status": "success",
            "filename": report.scan_details.filename,
            "timestamp": report.diagnosed_at.isoformat() + "Z",
            "traditional_model": {
                "prediction": report.traditional_prediction.prediction_label,
                "confidence": report.traditional_prediction.confidence_score,
                "features": report.traditional_prediction.extracted_features,
                "per_class_probabilities": rf_probabilities,
            },
            "cnn_model": {
                "prediction": report.cnn_prediction.prediction_label,
                "confidence": report.cnn_prediction.confidence_score,
                "grad_cam_path": report.cnn_prediction.grad_cam_path or "",
                "per_class_probabilities": cnn_probabilities,
            },
            "llm_narrative": report.llm_narrative,
            "session_title": report.session_title,
        }


class ChatController:
    """Controller to adapt chat JSON requests into LLM follow-up use cases.

    Assignee Guidelines:
    1. Parse user chat queries and histories.
    2. Convert session history entries into domain entities.
    3. Invoke chatbot use cases.
    4. Format chatbot replies for web serialization.
    """

    def __init__(self, chat_use_case: ChatWithAssistantUseCase):
        self.chat_use_case = chat_use_case

    def handle_message(
        self,
        raw_history: List[Dict[str, str]],
        new_message: str,
        diagnostic_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Task Assignee Implementation steps:
        1. Verify that new_message is not empty.
        2. Iterate over raw_history list and convert dictionaries into ChatMessage domain objects.
        3. Trigger chat_use_case.execute() passing the message history, current message, and diagnostics.
        4. Format the returned ChatMessage entity as a dictionary containing:
           - role
           - content
           - timestamp (ISO formatted string)
        5. Return the serialized dict wrapper.
        """
        if not new_message or not new_message.strip():
            return {"status": "error", "message": "A message is required."}

        history = [
            ChatMessage(role=item["role"], content=item["content"])
            for item in raw_history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        response = self.chat_use_case.execute(history, new_message.strip(), diagnostic_context)
        return {
            "status": "success",
            "response": {
                "role": response.role,
                "content": response.content,
                "timestamp": response.timestamp.isoformat() + "Z",
            },
        }
