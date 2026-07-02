"""
Pydantic models for API request/response schemas.
Updated to match the expected SHL conversation format from sample conversations.
"""
from typing import Optional, Literal

from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────

class Message(BaseModel):
    """A single chat message."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Incoming chat request with conversation history."""
    messages: list[Message]


# ── Data Models ─────────────────────────────────────────────────

class Assessment(BaseModel):
    """An SHL assessment/test solution."""
    name: str
    url: str
    test_type: str          # Abbreviated: A, P, K, B, S (Ability, Personality, Knowledge, Biodata, Simulation)
    keys: str               # Full category: "Knowledge & Skills", "Personality & Behavior", etc.
    duration: Optional[str] = None
    languages: Optional[str] = None
    remote_testing: str = "Yes"
    adaptive_testing: str = "No"
    description: str = ""


# ── Response Models ─────────────────────────────────────────────

class ChatResponse(BaseModel):
    """Structured response from the chat agent."""
    reply: str
    recommendations: list[Assessment] = Field(default_factory=list)
    end_of_conversation: bool = False
