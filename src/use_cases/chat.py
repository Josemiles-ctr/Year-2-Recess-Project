from typing import List, Dict, Any
from src.domain.entities import ChatMessage
from src.interfaces.gateways import LlmServiceGateway

class ChatWithAssistantUseCase:
    """Usecase to process user conversational follow-ups with LLM API.
    
    Assignee Guidelines:
    1. Coordinate follow-up chat messages with the LLM service.
    2. Format message objects and call gateways.
    3. Return structured ChatMessage entities to the callers.
    """
    
    def __init__(self, llm_gateway: LlmServiceGateway):
        self.llm_gateway = llm_gateway

    def execute(
        self,
        chat_history: List[ChatMessage],
        new_user_message: str,
        diagnostic_context: Dict[str, Any]
    ) -> ChatMessage:
        """Task Assignee Implementation steps:
        1. Pass historical ChatMessage structures and the new question to llm_gateway.chat_follow_up().
        2. Supply the active diagnostic_context dictionary as environmental context.
        3. Retrieve the response text from the gateway.
        4. Wrap the response string inside a new ChatMessage entity (role='assistant') and return it.
        """
        response = self.llm_gateway.chat_follow_up(chat_history, new_user_message, diagnostic_context)
        return ChatMessage(role="assistant", content=response)
