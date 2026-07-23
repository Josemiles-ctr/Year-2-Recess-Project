from dataclasses import dataclass, field
import datetime


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
