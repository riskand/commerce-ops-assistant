# app/main.py
from fastapi import FastAPI

from app.routes import ask, chat, docsearch, prompts

app = FastAPI(title="Commerce Ops")
app.include_router(ask.router)
app.include_router(chat.router)
app.include_router(docsearch.router)
app.include_router(prompts.router)
