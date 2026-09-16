# app/routes/ask.py
from fastapi import APIRouter
from pydantic import BaseModel

from app.loop import run_loop, TOOL_SCHEMAS

router = APIRouter(tags=["ask"])


class Ask(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    turns: int
    tool_calls: list[dict]


@router.post("/ask", response_model=Answer)
async def ask(req: Ask) -> Answer:
    messages = [{"role": "user", "content": req.question}]
    text, history = await run_loop(messages, TOOL_SCHEMAS)
    used = [b for m in history if isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_use"]
    return Answer(answer=text, turns=len(history), tool_calls=used)
