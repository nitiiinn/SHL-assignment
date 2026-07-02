"""
Simple guardrails — keyword and pattern-based safety checks.
No LLM calls, just fast regex/keyword matching.
"""
import re


# Patterns that suggest prompt injection
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)",
    r"you\s+are\s+now\s+(?:a|an)\s+",
    r"forget\s+(your|all)\s+(rules|instructions|training)",
    r"system\s*prompt",
    r"act\s+as\s+(if|though)",
    r"pretend\s+(you|to\s+be)",
    r"jailbreak",
    r"DAN\s+mode",
]

# Keywords suggesting legal/hiring advice requests
LEGAL_KEYWORDS = [
    "legal advice",
    "is it legal",
    "can i fire",
    "should i hire",
    "discriminat",
    "lawsuit",
    "sue",
    "labor law",
    "employment law",
    "wrongful termination",
]

# Off-topic indicators (clearly non-SHL topics)
OFF_TOPIC_PATTERNS = [
    r"(weather|recipe|joke|poem|story|song|movie|game)\b",
    r"(write|generate|create)\s+(me\s+)?(a\s+)?(code|script|essay|email)",
    r"(who\s+is|what\s+is\s+the\s+capital|how\s+to\s+cook)",
]


def check_safety(text: str) -> str | None:
    """
    Check if the user's message is safe to process.

    Returns:
        None if safe, or a refusal reason string if unsafe.
    """
    text_lower = text.lower().strip()

    # Check prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return "prompt_injection"

    # Check legal/hiring advice
    for keyword in LEGAL_KEYWORDS:
        if keyword in text_lower:
            return "legal_advice"

    # Check off-topic
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, text_lower):
            return "off_topic"

    return None
