from typing import List
from src.domain.entities import ChatMessage
from src.interfaces.gateways import LlmServiceGateway


class ChatWithAssistantUseCase:
    def __init__(self, llm_gateway: LlmServiceGateway):
        self.llm_gateway = llm_gateway

    def execute(
        self,
        chat_history: List[ChatMessage],
        new_user_message: str,
    ) -> ChatMessage:
        response = self.llm_gateway.chat_follow_up(
            chat_history, new_user_message
        )
        return ChatMessage(role="assistant", content=response)
