"""Unit tests for agent clarification gating and conversation limits."""

from app.agent import (
    MAX_CHAT_TURNS,
    build_comparison_message,
    build_grounded_recommendation_message,
    build_follow_up_question,
    build_retrieval_query,
    build_turn_cap_message,
    extract_compare_terms,
    infer_follow_up_intent,
    is_turn_cap_reached,
    parse_shortlist_directives,
    select_shortlist,
    should_ask_follow_up,
)
from app.guardrails import check_safety


def test_language_request_first_turn_triggers_clarification():
    messages = [
        {
            "role": "user",
            "content": (
                "We're hiring bilingual healthcare admin staff in South Texas - "
                "they handle patient records and need to be assessed in Spanish. "
                "HIPAA compliance is critical. What assessments work?"
            ),
        }
    ]
    results = [
        {"name": "HIPAA (Security)", "rrf_score": 0.1373},
        {"name": "Dependability and Safety Instrument (DSI)", "rrf_score": 0.1362},
    ]

    assert should_ask_follow_up(messages, results) is True
    question = build_follow_up_question(messages, results)
    assert "English-language knowledge tests" in question


def test_first_turn_follow_up_asks_for_missing_hiring_details():
    messages = [
        {
            "role": "user",
            "content": (
                "Here's the JD for an engineer we need to fill. Can you recommend "
                "an assessment battery?"
            ),
        }
    ]
    results = [
        {"name": "Java Test", "rrf_score": 0.1514},
        {"name": "Occupational Personality Questionnaire", "rrf_score": 0.1414},
    ]

    assert should_ask_follow_up(messages, results) is True
    question = build_follow_up_question(messages, results)
    assert "over-recommend" not in question
    assert "What level are you hiring for" in question
    assert "cognitive ability, technical skills, or personality and behavior" in question


def test_confident_first_turn_can_recommend():
    messages = [
        {
            "role": "user",
            "content": (
                "We're hiring plant operators for a chemical facility. Safety is "
                "absolute top priority - reliability, procedure compliance, never "
                "cutting corners. What do you recommend?"
            ),
        }
    ]
    results = [
        {"name": "Dependability and Safety Instrument (DSI)", "rrf_score": 0.1814},
        {"name": "Manufac. & Indust. - Safety & Dependability 8.0", "rrf_score": 0.1344},
    ]

    assert should_ask_follow_up(messages, results) is False


def test_sales_reskilling_request_can_recommend_on_first_turn():
    messages = [
        {
            "role": "user",
            "content": (
                "As part of our restructuring and annual talent audit, we need "
                "to re-skill our Sales organization. What solutions do you recommend?"
            ),
        }
    ]
    results = [
        {"name": "OPQ MQ Sales Report", "rrf_score": 0.3465},
        {"name": "Retail Sales and Service Simulation", "rrf_score": 0.3299},
        {"name": "Global Skills Assessment", "rrf_score": 0.3010},
    ]

    assert should_ask_follow_up(messages, results) is False


def test_turn_cap_is_enforced():
    messages = []
    for i in range(1, MAX_CHAT_TURNS + 1):
        messages.extend([
            {"role": "user", "content": f"Turn {i}"},
            {"role": "assistant", "content": f"Reply {i}"},
        ])

    assert is_turn_cap_reached(messages) is True
    assert "limit" in build_turn_cap_message().lower()


def test_incomplete_fourth_turn_does_not_hit_cap():
    messages = [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Reply 2"},
        {"role": "user", "content": "Turn 3"},
        {"role": "assistant", "content": "Reply 3"},
        {"role": "user", "content": "Turn 4"},
    ]

    assert is_turn_cap_reached(messages) is False


def test_grounded_recommendation_message_mentions_retrieved_skill_matches():
    messages = [
        {
            "role": "user",
            "content": (
                "We need a senior full-stack engineer with Java, Spring, AWS, "
                "Docker, SQL, and microservices experience."
            ),
        }
    ]
    assessments = [
        {
            "name": "Docker (New)",
            "description": "Measures Docker container, data management, performance, and swarm knowledge.",
            "duration": "25 minutes",
            "keys": ["Knowledge & Skills"],
            "job_levels": ["Mid-Professional", "Supervisor"],
            "remote_testing": "Yes",
        },
        {
            "name": "Spring (New)",
            "description": "Measures Spring framework knowledge for enterprise Java development.",
            "duration": "20 minutes",
            "keys": ["Knowledge & Skills"],
            "job_levels": ["Mid-Professional"],
            "remote_testing": "Yes",
        },
    ]

    message = build_grounded_recommendation_message(messages, assessments)

    assert "Docker (New)" in message
    assert "there is no Docker assessment" not in message
    assert "Spring (New)" in message
    assert "covers docker" in message.lower()


def test_grounded_recommendation_message_is_more_conversational_for_refinement():
    messages = [
        {
            "role": "user",
            "content": "We need a senior full-stack engineer with Java, Spring, AWS, Docker, SQL, and microservices experience.",
        },
        {
            "role": "assistant",
            "content": "Here are some SHL assessments to consider.",
        },
        {
            "role": "user",
            "content": "On Java - they'd be working on existing services, not greenfield. Is the Advanced level the right pick?",
        },
    ]
    assessments = [
        {
            "name": "Core Java (Advanced Level) (New)",
            "description": "Measures advanced Java knowledge for enterprise application development.",
            "duration": "13 minutes",
            "keys": ["Knowledge & Skills"],
            "job_levels": ["Mid-Professional", "Supervisor"],
            "remote_testing": "Yes",
        },
        {
            "name": "RESTful Web Services (New)",
            "description": "Measures REST service design and implementation knowledge.",
            "duration": "12 minutes",
            "keys": ["Knowledge & Skills"],
            "job_levels": ["Mid-Professional"],
            "remote_testing": "Yes",
        },
    ]

    message = build_grounded_recommendation_message(messages, assessments, intent="refine")

    assert "reasonable fit" in message.lower()
    assert "existing Java services" in message
    assert "Do you want me to keep this Java-focused" in message
    assert "across" not in message.lower()
    assert "already" not in message.lower()


def test_assessment_follow_up_routes_to_refine():
    messages = [
        {"role": "user", "content": "Recommend an assessment battery for a senior engineer."},
        {"role": "assistant", "content": "Here are some SHL assessments to consider."},
        {"role": "user", "content": "Do we really need Verify G+ on top of all the technical tests? Feels redundant."},
    ]

    assert infer_follow_up_intent(messages) == "refine"


def test_assessment_follow_up_routes_to_compare():
    messages = [
        {"role": "user", "content": "Recommend an assessment battery for a senior engineer."},
        {"role": "assistant", "content": "Here are some SHL assessments to consider."},
        {"role": "user", "content": "Which one is better, Docker (New) vs Amazon Web Services (AWS) Development (New)?"},
    ]

    assert infer_follow_up_intent(messages) == "compare"


def test_extract_compare_terms_handles_between_question():
    terms = extract_compare_terms("What's the difference between OPQ and OPQ MQ Sales Report?")
    assert terms == ["OPQ", "OPQ MQ Sales Report"]


def test_build_comparison_message_explains_family_vs_report():
    messages = [
        {"role": "user", "content": "Recommend something for sales hiring."},
        {"role": "assistant", "content": "Here are some SHL sales assessments."},
        {"role": "user", "content": "What's the difference between OPQ and OPQ MQ Sales Report?"},
    ]
    assessments = [
        {
            "name": "OPQ MQ Sales Report",
            "description": "This OPQ report provides a summary of natural style that is critical to sales success and can also use MQ motivation data.",
            "duration": "",
            "keys": ["Personality & Behavior"],
        },
        {
            "name": "Occupational Personality Questionnaire OPQ32r",
            "description": "A widely used workplace personality questionnaire for understanding behavioral style.",
            "duration": "25 minutes",
            "keys": ["Personality & Behavior"],
        },
    ]

    message = build_comparison_message(messages, assessments)

    assert "broader personality questionnaire family" in message
    assert "sales-specific report" in message
    assert "whether the sales report alone is enough" in message


def test_off_topic_request_still_triggers_guardrails():
    assert check_safety("Tell me a joke about databases") == "off_topic"


def test_parse_shortlist_directives_handles_add_and_drop():
    include_terms, exclude_terms = parse_shortlist_directives(
        "Add AWS and Docker. Drop REST - the API design signal will already come through in Spring."
    )

    assert include_terms == {"aws", "docker"}
    assert exclude_terms == {"rest"}


def test_select_shortlist_honors_add_and_drop_directives():
    results = [
        {"name": "Smart Interview Live Coding", "description": "Live coding interview.", "rrf_score": 0.40, "keys": ["Knowledge & Skills"]},
        {"name": "Docker (New)", "description": "Measures Docker knowledge.", "rrf_score": 0.35, "keys": ["Knowledge & Skills"]},
        {"name": "Spring (New)", "description": "Measures Spring framework knowledge.", "rrf_score": 0.34, "keys": ["Knowledge & Skills"]},
        {"name": "Amazon Web Services (AWS) Development (New)", "description": "Measures AWS development knowledge.", "rrf_score": 0.33, "keys": ["Knowledge & Skills"]},
        {"name": "RESTful Web Services (New)", "description": "Measures REST service design.", "rrf_score": 0.32, "keys": ["Knowledge & Skills"]},
    ]

    shortlist = select_shortlist(
        results,
        max_items=10,
        query_text="Add AWS and Docker. Drop REST - the API design signal will already come through in Spring.",
    )
    names = [item["name"] for item in shortlist]

    assert "Docker (New)" in names
    assert "Amazon Web Services (AWS) Development (New)" in names
    assert "RESTful Web Services (New)" not in names
