"""Simple deterministic chat handler for SHL assessment recommendations."""

import re

from app.guardrails import check_safety

MAX_CHAT_TURNS = 8


TYPE_ABBREVIATIONS = {
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Knowledge & Skills": "K",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)?", re.IGNORECASE)

GENERIC_QUERY_TERMS = {
    "a", "across", "add", "advanced", "all", "already", "an", "and", "api",
    "are", "as", "assessment", "assessments", "battery", "can", "cd",
    "cloud-native", "core", "delivery", "deployment", "design", "difference",
    "do", "does", "end", "engineer", "engineers", "existing", "feels", "fill",
    "for", "full", "greenfield", "here", "hiring", "how", "i", "in", "is",
    "it", "its", "jd", "level", "me", "need", "not", "of", "on", "one", "or",
    "our", "pick", "position", "really", "recommend", "redundant", "report",
    "required", "right", "role", "services", "should", "skills", "stack",
    "strong", "the", "they", "this", "to", "up", "use", "vs", "we", "what",
    "which", "with", "work", "working", "would", "years",
}

DIRECTIVE_TERM_RE = re.compile(
    r"\b(aws|docker|rest|spring|java|sql|angular|microservices?|verify\s*g\+|opq)\b",
    re.IGNORECASE,
)

LANGUAGE_REQUEST_RE = re.compile(
    r"\b(spanish|latin american spanish|french|german|portuguese|arabic|chinese|bilingual|multilingual)\b",
    re.IGNORECASE,
)

ENGLISH_FLUENCY_RE = re.compile(
    r"\b(english fluent|fluent in english|functionally bilingual|english is fine|english okay|can take english|written work in english)\b",
    re.IGNORECASE,
)

ASSESSMENT_FOLLOW_UP_RE = re.compile(
    r"\b("
    r"verify\s*g\+|verify|opq|assessment\s+battery|battery|shortlist|technical\s+tests?|"
    r"personality\s+test|cognitive\s+test|docker|aws|spring|java|sql|angular|"
    r"microservices?|redundant|too\s+many|drop|keep|remove|narrow|instead|"
    r"on\s+top\s+of|right\s+pick|compare|versus|vs\.?"
    r")\b",
    re.IGNORECASE,
)

COMPARE_HINT_RE = re.compile(
    r"\b(compare|versus|vs\.?|difference|better than|which one)\b",
    re.IGNORECASE,
)

PROGRAM_RECOMMENDATION_RE = re.compile(
    r"\b(re[-\s]?skill|upskill|talent audit|restructur|sales organization|sales team|annual talent)\b",
    re.IGNORECASE,
)

QUERY_EXPANSIONS = [
    (re.compile(r"\bmid[-\s]?level\b", re.IGNORECASE), "mid-professional"),
    (re.compile(r"\bjunior\b|\bentry[-\s]?level\b|\bintern\b|\bgraduate\b", re.IGNORECASE), "entry-level graduate"),
    (re.compile(r"\bsenior\b|\bexecutive\b|\bdirector\b|\blead\b", re.IGNORECASE), "senior executive director lead"),
    (re.compile(r"\bre[-\s]?skill\b|\bupskill\b|\btalent audit\b|\brestructur", re.IGNORECASE), "global skills assessment global skills development report development 360 reskilling"),
    (re.compile(r"\bsales organization\b|\bsales team\b|\bsalesforce\b|\bsales\b", re.IGNORECASE), "sales transformation sales manager individual contributor opq personality"),
    (re.compile(r"\bpersonality\b|\bbehavior\b", re.IGNORECASE), "personality behavior"),
    (re.compile(r"\bcognitive\b|\bability\b|\baptitude\b|\breasoning\b", re.IGNORECASE), "ability aptitude cognitive reasoning"),
    (re.compile(r"\btechnical\b|\bskills\b|\bknowledge\b|\bcoding\b|\bdeveloper\b|\bjava\b|\bpython\b|\bnet\b", re.IGNORECASE), "knowledge skills technical coding"),
    (re.compile(r"\bdocker\b|\baws\b|\bangular\b|\bspring\b|\bsql\b|\brest(?:ful)?\b|\bmicroservices?\b|\bcloud\b|\bci/cd\b|\bcicd\b", re.IGNORECASE), "docker aws angular spring sql restful microservices cloud cicd devops"),
    (re.compile(r"\bstakeholder\b|\bcommunication\b|\bcollaborat", re.IGNORECASE), "stakeholder communication collaboration interpersonal"),
    (re.compile(r"\bremote\b|\badaptive\b", re.IGNORECASE), "remote adaptive"),
]


retriever = None


def get_type_abbreviation(keys: list) -> str:
    if not keys:
        return "K"
    first_key = keys[0] if isinstance(keys, list) else str(keys)
    return TYPE_ABBREVIATIONS.get(first_key, "K")


def format_languages(languages: list) -> str:
    if not languages:
        return "-"
    if len(languages) <= 3:
        return ", ".join(languages)
    shown = ", ".join(languages[:3])
    return f"{shown} _(+{len(languages) - 3} more)_"


def _tokenize_text(text: str) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def _extract_priority_terms(query_text: str) -> set[str]:
    priority_terms = set()
    for token in _tokenize_text(query_text):
        if len(token) <= 2 or token in GENERIC_QUERY_TERMS or token.isdigit():
            continue
        priority_terms.add(token)
    return priority_terms


def _latest_user_message(messages: list[dict]) -> str:
    return next(
        (msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"),
        "",
    )


def _normalize_directive_term(term: str) -> str:
    normalized = term.lower().strip()
    if normalized.startswith("verify"):
        return "verify g+"
    if normalized == "microservice":
        return "microservices"
    return normalized


def parse_shortlist_directives(query_text: str) -> tuple[set[str], set[str]]:
    includes: set[str] = set()
    excludes: set[str] = set()
    text = query_text or ""

    for match in re.finditer(r"(?:add|include|keep|prioritize)\s+([^.;\n\-]+)", text, re.IGNORECASE):
        for term in DIRECTIVE_TERM_RE.findall(match.group(1)):
            includes.add(_normalize_directive_term(term))

    for match in re.finditer(r"(?:drop|remove|exclude)\s+([^.;\n\-]+)", text, re.IGNORECASE):
        for term in DIRECTIVE_TERM_RE.findall(match.group(1)):
            excludes.add(_normalize_directive_term(term))

    return includes, excludes


def _assessment_text(assessment: dict) -> str:
    return " ".join(
        [
            assessment.get("name", ""),
            assessment.get("description", ""),
            " ".join(assessment.get("keys", []) or []),
        ]
    ).lower()


def _assessment_matches_terms(assessment: dict, terms: set[str]) -> bool:
    if not terms:
        return False
    text = _assessment_text(assessment)
    tokens = _tokenize_text(text)
    return any(term in text or term in tokens for term in terms)


def select_shortlist(results: list[dict], max_items: int = 10, query_text: str = "") -> list[dict]:
    if not results:
        return []

    include_terms, exclude_terms = parse_shortlist_directives(query_text)
    filtered_results = [
        result for result in results if not _assessment_matches_terms(result, exclude_terms)
    ]
    if filtered_results:
        results = filtered_results

    shortlisted = [results[0]]
    top_score = float(results[0].get("rrf_score", 0.0))
    if len(results) == 1:
        return shortlisted

    relative_floor = top_score * 0.7
    absolute_floor = max(top_score * 0.45, 0.05)
    max_adaptive = 5

    for result in results[1:min(max_items, max_adaptive)]:
        score = float(result.get("rrf_score", 0.0))
        if score < absolute_floor or score < relative_floor:
            break
        shortlisted.append(result)

    priority_terms = _extract_priority_terms(query_text)
    if priority_terms and len(shortlisted) < max_items:
        shortlisted_names = {item.get("name", "") for item in shortlisted}
        for result in results[:max_items]:
            if result.get("name", "") in shortlisted_names:
                continue
            if float(result.get("rrf_score", 0.0)) < absolute_floor:
                continue
            if not (_tokenize_text(result.get("name", "")) & priority_terms):
                continue
            shortlisted.append(result)
            shortlisted_names.add(result.get("name", ""))
            if len(shortlisted) >= max_items:
                break

    if include_terms and len(shortlisted) < max_items:
        shortlisted_names = {item.get("name", "") for item in shortlisted}
        for result in results:
            if result.get("name", "") in shortlisted_names:
                continue
            if not _assessment_matches_terms(result, include_terms):
                continue
            shortlisted.append(result)
            shortlisted_names.add(result.get("name", ""))
            if len(shortlisted) >= max_items:
                break

    return shortlisted


def _expand_query_text(text: str) -> str:
    expanded = text.strip()
    for pattern, replacement in QUERY_EXPANSIONS:
        if pattern.search(expanded):
            expanded = f"{expanded} {replacement}"
    return expanded


def build_retrieval_query(messages: list[dict]) -> str:
    user_messages = [
        msg["content"].strip()
        for msg in messages
        if msg.get("role") == "user" and msg.get("content")
    ]
    if not user_messages:
        return ""

    latest_user = user_messages[-1]
    recent_context = " ".join(user_messages[-3:-1]) if len(user_messages) > 1 else ""
    all_user_text = " ".join(user_messages)
    expanded = _expand_query_text(all_user_text)
    return " \n".join(part for part in [latest_user, recent_context, expanded] if part)


def infer_follow_up_intent(messages: list[dict]) -> str | None:
    user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    if not user_messages or not any(msg.get("role") == "assistant" for msg in messages):
        return None

    latest_user = user_messages[-1]
    if not ASSESSMENT_FOLLOW_UP_RE.search(latest_user):
        return None
    if COMPARE_HINT_RE.search(latest_user):
        return "compare"
    return "refine"


def should_recommend_on_first_turn(latest_user: str, results: list[dict]) -> bool:
    if not latest_user or not results:
        return False
    if PROGRAM_RECOMMENDATION_RE.search(latest_user):
        return True

    top_score = float(results[0].get("rrf_score", 0.0))
    if top_score >= 0.17 and any(term in latest_user.lower() for term in ["sales", "organization", "team"]):
        return True
    return False


def should_ask_follow_up(messages: list[dict], results: list[dict]) -> bool:
    user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    if not user_messages:
        return True

    latest_user = user_messages[-1]
    first_turn = len(user_messages) == 1
    if not first_turn:
        return False

    if LANGUAGE_REQUEST_RE.search(latest_user) and not ENGLISH_FLUENCY_RE.search(latest_user):
        return True
    if not results:
        return True
    if should_recommend_on_first_turn(latest_user, results):
        return False
    if len(results) >= 2:
        top_score = float(results[0].get("rrf_score", 0.0))
        second_score = float(results[1].get("rrf_score", 0.0))
        if (top_score - second_score) < 0.03:
            return True
    return False


def build_follow_up_question(messages: list[dict], results: list[dict]) -> str:
    latest_user = _latest_user_message(messages)
    if LANGUAGE_REQUEST_RE.search(latest_user) and not ENGLISH_FLUENCY_RE.search(latest_user):
        return (
            "Are your candidates fluent enough to take English-language knowledge tests, "
            "or do you need all assessments to be available in Spanish?"
        )
    if results:
        return (
            "What level are you hiring for, and which assessment focus should I prioritize: "
            "cognitive ability, technical skills, or personality and behavior?"
        )
    return (
        "I need a bit more detail before I recommend assessments. "
        "What role are you hiring for, and what skills or constraints matter most?"
    )


def is_turn_cap_reached(messages: list[dict], max_turns: int = MAX_CHAT_TURNS) -> bool:
    completed_turns = min(
        sum(1 for msg in messages if msg.get("role") == "user"),
        sum(1 for msg in messages if msg.get("role") == "assistant"),
    )
    return completed_turns >= max_turns


def build_turn_cap_message() -> str:
    return (
        "We have reached the conversation limit for this session. "
        "Please start a new request if you want to continue."
    )


def _extract_level_hint(text: str) -> str | None:
    text_lower = (text or "").lower()
    level_map = [
        ("entry-level", ["entry", "junior", "graduate", "intern"]),
        ("mid-level", ["mid", "mid-level", "midlevel"]),
        ("senior-level", ["senior", "lead", "principal"]),
        ("executive-level", ["director", "executive", "vp", "head of"]),
    ]
    for label, keywords in level_map:
        if any(keyword in text_lower for keyword in keywords):
            return label
    return None


def _build_fit_reason(assessment: dict, query_text: str) -> str:
    priority_terms = _extract_priority_terms(query_text)
    skill_hits = sorted((_tokenize_text(assessment.get("name", "")) | _tokenize_text(assessment.get("description", ""))) & priority_terms)[:3]
    if skill_hits:
        return f"it covers {', '.join(skill_hits)}"

    keys = assessment.get("keys", [])
    keys_str = ", ".join(keys) if isinstance(keys, list) else str(keys)
    if "Knowledge & Skills" in keys_str:
        return "it matches the technical screening focus"
    if "Ability & Aptitude" in keys_str:
        return "it supports cognitive screening for the role"
    if "Personality & Behavior" in keys_str:
        return "it helps evaluate behavioral fit"
    return "it is one of the closest SHL matches for the skills you described"


def extract_compare_terms(text: str) -> list[str]:
    text = (text or "").strip()
    patterns = [
        r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"compare\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
        r"(.+?)\s+versus\s+(.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return [match.group(1).strip(), match.group(2).strip()]
    return []


def summarize_assessment_focus(assessment: dict) -> str:
    description = assessment.get("description", "").lower()
    name = assessment.get("name", "").lower()
    if "sales success" in description or "sales" in name:
        return "a sales-focused personality or development view"
    if "personality questionnaire" in description or "personality" in description:
        return "a broad workplace personality measure"
    if "development" in description:
        return "a development-oriented report"
    if "manager" in description:
        return "a manager-facing interpretation report"
    if "simulation" in description:
        return "a simulation-based assessment"
    return "a targeted SHL assessment for this area"


def _best_match_for_term(term: str, assessments: list[dict]) -> dict | None:
    term_tokens = _tokenize_text(term)
    best_match = None
    best_score = -1
    for assessment in assessments:
        score = len(term_tokens & _tokenize_text(_assessment_text(assessment)))
        if term.lower() in assessment.get("name", "").lower():
            score += 3
        if score > best_score:
            best_score = score
            best_match = assessment
    return best_match if best_score > 0 else None


def build_comparison_message(messages: list[dict], assessments: list[dict]) -> str:
    latest_user = _latest_user_message(messages)
    compare_terms = extract_compare_terms(latest_user)

    if len(compare_terms) == 2:
        left_term, right_term = compare_terms
        if "opq" in left_term.lower() and "sales" in right_term.lower():
            sales_report = next(
                (a for a in assessments if "opq mq sales report" in a.get("name", "").lower()),
                None,
            )
            report_name = sales_report.get("name", "OPQ MQ Sales Report") if sales_report else "OPQ MQ Sales Report"
            return (
                f"The main difference is that OPQ is the broader personality questionnaire family, "
                f"while {report_name} is a sales-specific report built from OPQ results and, where used, MQ motivation data.\n\n"
                f"Use OPQ when you want a general view of workplace behavioral style across roles. "
                f"Use {report_name} when you want that personality signal translated into sales-relevant strengths, risks, and motivators.\n\n"
                "If you want, I can also tell you when the sales report adds signal on top of the base OPQ and whether the sales report alone is enough."
            )

        left = _best_match_for_term(left_term, assessments)
        right = _best_match_for_term(right_term, assessments)
        if left and right and left.get("name") != right.get("name"):
            left_name = left.get("name", left_term)
            right_name = right.get("name", right_term)
            left_focus = summarize_assessment_focus(left)
            right_focus = summarize_assessment_focus(right)
            return (
                f"Here’s the practical difference. {left_name} is better when you want {left_focus}, "
                f"while {right_name} is better when you want {right_focus}.\n\n"
                f"If the role needs stronger emphasis on {left_term}, I’d lean toward {left_name}. "
                f"If the priority is {right_term}, I’d lean toward {right_name}.\n\n"
                "If you want, I can turn that into a simple 'use this when / skip this when' recommendation."
            )

    if len(assessments) >= 2:
        first, second = assessments[0], assessments[1]
        return (
            f"The main difference is that {first.get('name', 'Option 1')} is {summarize_assessment_focus(first)}, "
            f"while {second.get('name', 'Option 2')} is {summarize_assessment_focus(second)}.\n\n"
            "If you want, I can compare them side by side on purpose, overlap, and whether you need both."
        )

    return (
        "I can compare the options, but I need two concrete SHL assessments in the shortlist first. "
        "If you name the two assessments, I’ll give you a direct side-by-side answer."
    )


def _build_reply_intro(intent: str, latest_user: str, assessments: list[dict]) -> str:
    latest_lower = latest_user.lower()
    first_name = assessments[0].get("name", "the leading option") if assessments else "the leading option"

    if intent == "compare":
        return "Here’s how I’d compare the closest SHL options for that follow-up."
    if "redundant" in latest_lower or "too many" in latest_lower:
        return "Yes, that can be redundant. I’d streamline the battery rather than stack overlapping tests."
    if "right pick" in latest_lower or "advanced level" in latest_lower:
        return f"For someone working on existing Java services, {first_name} is a reasonable fit if you want hands-on knowledge rather than only general aptitude."
    if "drop" in latest_lower or "remove" in latest_lower:
        return "Yes, we can trim this down and keep only the assessments that add distinct signal."
    return "Based on what you’ve shared, I’d narrow the SHL shortlist to these options."


def _build_follow_up_prompt(intent: str, latest_user: str, assessments: list[dict]) -> str:
    latest_lower = latest_user.lower()
    if intent == "compare":
        return "If you want, I can also give you a simple recommendation on which one to keep."
    if "advanced level" in latest_lower or "right pick" in latest_lower:
        return "Do you want me to keep this Java-focused, or add one broader screen for problem-solving or behavioral fit?"
    if "redundant" in latest_lower or "too many" in latest_lower:
        return "If you want, I can shrink this to a lean 3-test battery."
    if len(assessments) >= 3:
        return "If you want, I can split these into must-have versus optional assessments."
    return "If you want, I can refine this further once you tell me which signal matters most."


def build_grounded_recommendation_message(messages: list[dict], assessments: list[dict], intent: str = "recommend") -> str:
    if not assessments:
        return (
            "I couldn't find a confident SHL shortlist from the details so far. "
            "If you share the role level and the top technical or behavioral skills, I can narrow it down."
        )

    if intent == "compare":
        return build_comparison_message(messages, assessments)

    latest_user = _latest_user_message(messages)
    search_query = latest_user or build_retrieval_query(messages)
    level_hint = _extract_level_hint(search_query)
    opener = _build_reply_intro(intent, latest_user, assessments)
    if intent == "recommend" and level_hint:
        opener = f"For this {level_hint} role, I’d prioritize these SHL assessments."

    shortlist = assessments[:3] if intent == "refine" else assessments[:5]
    lines = [opener, ""]
    for index, assessment in enumerate(shortlist, start=1):
        duration = assessment.get("duration") or "not specified"
        lines.append(
            f"{index}. {assessment.get('name', 'Unknown')} ({duration}): {_build_fit_reason(assessment, search_query)}."
        )

    priority_terms = _extract_priority_terms(search_query)
    covered_terms = set()
    for assessment in shortlist:
        covered_terms |= _tokenize_text(assessment.get("name", ""))
        covered_terms |= _tokenize_text(assessment.get("description", ""))

    uncovered_terms = [
        term for term in sorted(priority_terms)
        if term not in covered_terms and term not in {"cicd", "restful"}
    ]
    if uncovered_terms:
        lines.extend(
            [
                "",
                "I didn’t see dedicated SHL tests in this shortlist for "
                f"{', '.join(uncovered_terms[:2])}, so I prioritized the closest stack-relevant options instead.",
            ]
        )

    lines.extend(["", _build_follow_up_prompt(intent, latest_user, shortlist)])
    return "\n".join(lines)


def _build_refusal_message(reason: str) -> str:
    if reason == "prompt_injection":
        return "I can help with SHL assessment recommendations, but I can’t follow instruction-bypassing or prompt-injection requests."
    if reason == "legal_advice":
        return "I can help you choose SHL assessments, but I can’t provide legal or employment-law advice."
    return "I can help with SHL assessment recommendations and assessment-selection questions. If you share the role or talent objective, I’ll narrow the right options."


def _detect_intent(messages: list[dict]) -> str:
    follow_up_intent = infer_follow_up_intent(messages)
    if follow_up_intent:
        return follow_up_intent

    latest_user = _latest_user_message(messages).lower()
    if COMPARE_HINT_RE.search(latest_user):
        return "compare"
    return "recommend"


def _to_assessment_payload(assessments: list[dict]) -> list[dict]:
    payload = []
    for assessment in assessments:
        keys = assessment.get("keys", [])
        keys_str = ", ".join(keys) if isinstance(keys, list) else str(keys)
        payload.append(
            {
                "name": assessment.get("name", "Unknown"),
                "url": assessment.get("url", ""),
                "test_type": get_type_abbreviation(keys),
                "keys": keys_str,
                "duration": assessment.get("duration") or "-",
                "languages": format_languages(assessment.get("languages", [])),
                "remote_testing": assessment.get("remote_testing", "Yes"),
                "adaptive_testing": assessment.get("adaptive_testing", "No"),
                "description": assessment.get("description", ""),
            }
        )
    return payload


def handle_chat(messages: list[dict]) -> dict:
    if is_turn_cap_reached(messages):
        return {
            "response_type": "clarification",
            "message": build_turn_cap_message(),
            "assessments": [],
            "end_of_conversation": True,
        }

    latest_user = _latest_user_message(messages)
    if not latest_user:
        return {
            "response_type": "clarification",
            "message": "What role or talent goal should I help you with?",
            "assessments": [],
            "end_of_conversation": False,
        }

    safety_issue = check_safety(latest_user)
    if safety_issue:
        return {
            "response_type": "refusal",
            "message": _build_refusal_message(safety_issue),
            "assessments": [],
            "end_of_conversation": True,
        }

    intent = _detect_intent(messages)
    search_query = build_retrieval_query(messages)
    results = retriever.search(search_query, top_k=10) if retriever is not None else []

    if should_ask_follow_up(messages, results):
        return {
            "response_type": "clarification",
            "message": build_follow_up_question(messages, results),
            "assessments": [],
            "end_of_conversation": False,
        }

    shortlist = select_shortlist(results, max_items=10, query_text=search_query)
    response_type = "comparison" if intent == "compare" else "refinement" if intent == "refine" else "recommendation"
    return {
        "response_type": response_type,
        "message": build_grounded_recommendation_message(messages, shortlist, intent=intent),
        "assessments": _to_assessment_payload(shortlist),
        "end_of_conversation": False,
    }
