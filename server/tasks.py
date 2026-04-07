"""
tasks.py — ticket bank and grader logic for the Customer Support RL environment.

Each task is a dict with:
    ticket_id         : unique ID
    difficulty        : 'easy' | 'medium' | 'hard'
    customer_message  : raw text the agent receives
    correct_action    : canonical action_type that should be chosen
    expected_keywords : words the response should include for full marks
    max_steps         : how many agent turns are allowed
"""

import re

TASKS = [
    # ── EASY (clear-cut, single correct action) ───────────────────────────────
    {
        "ticket_id": "T001",
        "difficulty": "easy",
        "customer_message": (
            "Hi, I was charged twice for my subscription this month. "
            "I need my money back as soon as possible."
        ),
        "correct_action": "refund",
        "expected_keywords": ["refund", "sorry", "charge", "return"],
        "max_steps": 3,
    },
    {
        "ticket_id": "T002",
        "difficulty": "easy",
        "customer_message": (
            "I forgot my password. How do I reset it?"
        ),
        "correct_action": "inform",
        "expected_keywords": ["reset", "password", "link", "email"],
        "max_steps": 3,
    },
    {
        "ticket_id": "T003",
        "difficulty": "easy",
        "customer_message": (
            "My app keeps crashing every time I open it on my iPhone."
        ),
        "correct_action": "troubleshoot",
        "expected_keywords": ["restart", "update", "reinstall", "steps"],
        "max_steps": 3,
    },

    # ── MEDIUM (requires interpretation, partial ambiguity) ───────────────────
    {
        "ticket_id": "T004",
        "difficulty": "medium",
        "customer_message": (
            "I cancelled my plan three days ago but I was still charged today. "
            "I've already tried contacting billing twice and nobody helped me. "
            "This is completely unacceptable."
        ),
        "correct_action": "refund",
        "expected_keywords": ["apologize", "refund", "billing", "investigate"],
        "max_steps": 4,
    },
    {
        "ticket_id": "T005",
        "difficulty": "medium",
        "customer_message": (
            "I think there might be unauthorized access to my account. "
            "I received a login alert from a country I've never been to."
        ),
        "correct_action": "escalate",
        "expected_keywords": ["security", "team", "escalate", "urgent", "lock"],
        "max_steps": 4,
    },
    {
        "ticket_id": "T006",
        "difficulty": "medium",
        "customer_message": (
            "The dashboard is loading very slowly and some charts aren't showing. "
            "It started after your maintenance window yesterday."
        ),
        "correct_action": "troubleshoot",
        "expected_keywords": ["cache", "browser", "refresh", "status", "issue"],
        "max_steps": 4,
    },

    # ── HARD (complex, multi-issue, requires nuanced reasoning) ───────────────
    {
        "ticket_id": "T007",
        "difficulty": "hard",
        "customer_message": (
            "I run a small business and your API has been down for 6 hours. "
            "We've lost thousands of dollars. I need a refund AND I need to speak "
            "to someone senior. Your status page still says everything is operational "
            "which is clearly wrong. My team is furious."
        ),
        "correct_action": "escalate",
        "expected_keywords": ["escalate", "senior", "apologize", "outage", "compensate", "team"],
        "max_steps": 5,
    },
    {
        "ticket_id": "T008",
        "difficulty": "hard",
        "customer_message": (
            "Your AI feature gave my client completely wrong financial advice. "
            "We used it in good faith and now we might face legal action. "
            "I need to understand your liability policy and I want all logs "
            "of our session exported immediately."
        ),
        "correct_action": "escalate",
        "expected_keywords": ["legal", "escalate", "logs", "team", "sorry", "policy"],
        "max_steps": 5,
    },
    {
        "ticket_id": "T009",
        "difficulty": "hard",
        "customer_message": (
            "I'm a developer. I've followed all your docs but the OAuth2 flow "
            "keeps returning a 401 even though my credentials are correct. "
            "I've tested on Postman and it works, but not in my Next.js app. "
            "I need help debugging the exact difference."
        ),
        "correct_action": "troubleshoot",
        "expected_keywords": ["headers", "token", "cors", "redirect", "debug", "oauth"],
        "max_steps": 5,
    },
]


# ── Grader ────────────────────────────────────────────────────────────────────

def grade(task: dict, action_type: str, response: str) -> tuple[float, str]:
    """
    Return (reward: float 0.0–1.0, feedback: str).

    Scoring breakdown:
        0.50  — correct action_type chosen
        0.30  — keyword coverage in the response
        0.20  — response quality (length, politeness, clarity)
    """
    reward = 0.0
    notes = []

    # 1. Action type correctness (0.5)
    if action_type == task["correct_action"]:
        reward += 0.50
        notes.append("✓ Correct action type selected (+0.50)")
    else:
        notes.append(
            f"✗ Wrong action '{action_type}', expected '{task['correct_action']}' (+0.00)"
        )

    # 2. Keyword coverage (0.30)
    response_lower = response.lower()
    keywords_hit = [kw for kw in task["expected_keywords"] if kw in response_lower]
    kw_ratio = len(keywords_hit) / max(len(task["expected_keywords"]), 1)
    kw_score = round(kw_ratio * 0.30, 4)
    reward += kw_score
    notes.append(
        f"Keywords matched: {keywords_hit} → +{kw_score:.2f}"
    )

    # 3. Response quality (0.20)
    quality_score = 0.0
    word_count = len(response.split())

    # Length signal: 20-150 words is ideal
    if word_count >= 20:
        quality_score += 0.08
    if word_count >= 50:
        quality_score += 0.04

    # Politeness signals
    polite_words = ["sorry", "apologize", "thank", "understand", "appreciate", "please"]
    if any(pw in response_lower for pw in polite_words):
        quality_score += 0.05

    # No aggressive / dismissive language
    bad_phrases = ["not my problem", "you're wrong", "impossible", "never", "don't care"]
    if not any(bp in response_lower for bp in bad_phrases):
        quality_score += 0.03

    quality_score = round(min(quality_score, 0.20), 4)
    reward += quality_score
    notes.append(f"Quality score: +{quality_score:.2f} (words={word_count})")

    reward = round(min(reward, 1.0), 4)
    feedback = " | ".join(notes)
    return reward, feedback
