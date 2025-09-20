from pydantic import BaseModel
from .models import MessageRole


class ThreadUpdateSchema(BaseModel):
    new_title: str


class MessageCreateSchema(BaseModel):
    content: str


class ThreadCreateSchema(BaseModel):
    title: str


class ThreadSchema(BaseModel):
    id: str
    title: str
    messages: list["MessageSchema"] = []


class MessageSchema(BaseModel):
    id: str
    content: str
    role: MessageRole
    thread_id: str
