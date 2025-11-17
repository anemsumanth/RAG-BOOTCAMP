################### FastAPI backend

# app/main.py
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from .graph import graph, AgentState
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="RAG‑ReAct Agent", version="1.0")

# CORS for local dev + simple UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[Dict[str, Any]] | None = None

# ----------------------------------------------------
# Helper: call the graph once per request
# ----------------------------------------------------
def run_agent(message: str) -> ChatResponse:
    # FastAPI creates a fresh request‑scope object each call
    state: AgentState = {"messages": [HumanMessage(content=message)]}
    result = graph.invoke(state)

    # Extract last AI message
    ai_msg: AIMessage = result["messages"][-1]
    return ChatResponse(answer=ai_msg.content)

# ----------------------------------------------------
# Endpoints
# ----------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        return run_agent(req.message)
    except Exception as exc:
        # In production you’d log this!
        raise HTTPException(status_code=500, detail=str(exc))

# Health check
@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})