from typing import List, Dict, Any
from src.domain.entities import PredictionResult, ChatMessage
from src.interfaces.gateways import LlmServiceGateway

class GeminiLlmService(LlmServiceGateway):
    """Large Language Model Integration.
    
    Assignee Guidelines:
    1. Connect to an LLM provider (e.g. Google Gemini API or OpenAI API) using environment variables.
    2. Package ML model diagnostics (verdict, probability, contrast, textures) into a prompt template.
    3. Generate detailed clinical narratives summarizing the findings.
    4. Implement multi-turn chat support injecting historical messages and the active diagnostic context.
    """

    def __init__(self):
        # TODO: Load LLM API keys and provider selection from environment configuration.
        # Initialize generative AI clients (e.g., google.generativeai or openai).
        pass

    def generate_report_narrative(
        self, 
        traditional_result: PredictionResult, 
        cnn_result: PredictionResult
    ) -> str:
        """Task Assignee Implementation steps:
        1. Parse predictions and scores from traditional_result and cnn_result.
        2. Format a system instruction directing the model to act as a clinical explainable AI.
        3. Compile a prompt summarizing the texture features (GLCM) and neural confidence outputs.
        4. Trigger the API call to generate a structured medical summary.
        5. Handle connection timeouts or missing key exceptions gracefully, returning a baseline fallback text.
        """
        # TODO: Implement diagnostic narrative generation.
        raise NotImplementedError("LLM Report Narrative Generation is not implemented yet.")

    def chat_follow_up(
        self, 
        history: List[ChatMessage], 
        new_message: str, 
        diagnostic_context: Dict[str, Any]
    ) -> str:
        """Task Assignee Implementation steps:
        1. Compile conversation history from the history list into the target API format (user/assistant turns).
        2. Formulate a system prompt containing the active patient's diagnostic context (predictions, confidence metrics).
        3. Append the new_message as the latest user turn.
        4. Invoke the API to generate a context-aware chatbot response.
        5. Return the resulting text string.
        """
        # TODO: Implement chatbot message processing.
        raise NotImplementedError("LLM Chat Follow-Up is not implemented yet.")
