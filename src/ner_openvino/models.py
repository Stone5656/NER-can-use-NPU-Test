from typing import List, Optional
from datetime import datetime, timezone
from sqlmodel import Field, Relationship, SQLModel
from enum import Enum


class MessageRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"


class AiThread(SQLModel, table=True):
    id: str = Field(primary_key=True)
    title: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)

    messages: List["AiMessage"] = Relationship(back_populates="thread")


class AiMessage(SQLModel, table=True):
    id: str = Field(primary_key=True)
    content: str
    role: MessageRole
    thread_id: str = Field(foreign_key="aithread.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)

    thread: Optional[AiThread] = Relationship(back_populates="messages")
