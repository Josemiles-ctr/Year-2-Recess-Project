from abc import ABC, abstractmethod
from typing import List
from src.domain.entities import ChatMessage


class LlmServiceGateway(ABC):
    @abstractmethod
    def chat_follow_up(
        self, history: List[ChatMessage], new_message: str
    ) -> str:
        pass
