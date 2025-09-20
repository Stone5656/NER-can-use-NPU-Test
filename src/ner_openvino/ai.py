import os
from uuid import uuid4

from google import genai
from google.genai import types
from sqlmodel import Session, select

from .models import AiMessage, AiThread, MessageRole

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


async def add_response(thread: AiThread, session: Session) -> AiMessage:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    msgs = session.exec(
        select(AiMessage)
        .where(AiMessage.thread_id == thread.id)
        .order_by(AiMessage.created_at)
    ).all()
    contents = []
    for m in msgs:
        role = "user" if m.role == MessageRole.USER else "model"
        contents.append(
            types.Content(role=role, parts=[
                types.Part.from_text(text=m.content)])
        )
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
        system_instruction=[
            types.Part.from_text(
                text="""You are an incredibly helpful and friendly assistant."""),  # TODO: system instructionは要調整
        ],
    )

    ai_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=generate_content_config,
    )
    response_message = AiMessage(
        id=str(uuid4()),
        content=ai_response.text,
        role=MessageRole.ASSISTANT,
        thread_id=thread.id,
    )
    session.add(response_message)
    session.commit()
    session.refresh(response_message)
    return response_message
