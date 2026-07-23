from typing import Dict, Any, List
from src.domain.entities import ChatMessage
from src.use_cases.chat import ChatWithAssistantUseCase


class ChatController:
    def __init__(self, chat_use_case: ChatWithAssistantUseCase):
        self.chat_use_case = chat_use_case

    def handle_message(
        self,
        raw_history: List[Dict[str, str]],
        new_message: str,
    ) -> Dict[str, Any]:
        if not new_message or not new_message.strip():
            return {"status": "error", "message": "A message is required."}

        history = [
            ChatMessage(role=item["role"], content=item["content"])
            for item in raw_history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        response = self.chat_use_case.execute(history, new_message.strip())
        return {
            "status": "success",
            "response": {
                "role": response.role,
                "content": response.content,
                "timestamp": response.timestamp.isoformat() + "Z",
            },
        }
