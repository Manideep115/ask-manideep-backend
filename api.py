from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from query_engine import process_query

app = FastAPI(
    title="Ask Manideep API",
    description="AI-powered portfolio assistant for Alur Manideep",
    version="1.0.0"
)

# CORS — allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConversationContext(BaseModel):
    last_project: Optional[str] = None
    last_domain: Optional[str] = None
    last_topic: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    conversation_context: Optional[ConversationContext] = None


class ChatResponse(BaseModel):
    answer: str
    context_used: list[str]


@app.get("/")
async def root():
    return {"status": "online", "service": "Ask Manideep API"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(request.query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 chars)")

    ctx = None
    if request.conversation_context:
        ctx = request.conversation_context.model_dump(exclude_none=True)

    result = await process_query(
        query=request.query.strip(),
        conversation_context=ctx
    )

    return ChatResponse(
        answer=result["answer"],
        context_used=result["context_used"]
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
