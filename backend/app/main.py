"""
FastAPI application — two endpoints: /health and /chat.
Loads retrieval indexes at startup.
"""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, ChatResponse
from app import agent as agent_module
from app.retrieval import load_retriever

load_dotenv()


# ── Startup / Shutdown ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load retrieval indexes when the server starts."""
    print("Loading retrieval indexes...")
    try:
        retriever = load_retriever()
        agent_module.retriever = retriever  # inject into agent module
        print(f"Loaded {len(retriever.assessments)} assessments.")
    except Exception as e:
        print(f"Warning: Could not load indexes: {e}")
        print("Server will run without retrieval (for testing).")
    yield
    print("Shutting down.")


# ── App Setup ───────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL Individual Test Solutions",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat endpoint — accepts conversation history, returns structured response.

    The agent is stateless: full conversation history is sent with each request.
    """
    # Convert messages to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if agent_module.is_turn_cap_reached(messages):
        return ChatResponse(
            reply=agent_module.build_turn_cap_message(),
            recommendations=[],
            end_of_conversation=True,
        )

    # Run the chat handler
    response_data = agent_module.handle_chat(messages)

    return ChatResponse(
        reply=response_data.get("message", "Sorry, something went wrong."),
        recommendations=response_data.get("assessments") or [],
        end_of_conversation=response_data.get("end_of_conversation", False),
    )
