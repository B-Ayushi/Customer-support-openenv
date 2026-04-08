"""
tasks.py — ticket bank and grader logic.
ALL rewards are strictly clamped to (0.01, 0.99) — never 0.0 or 1.0.
"""

TASKS = [
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
        "customer_message": "I forgot my password. How do I reset it?",
        "correct_action": "inform",
        "expected_keywords": ["reset", "password", "link", "email"],
        "max_steps": 3,
    },
    {
        "ticket_id": "T003",
        "difficulty": "easy",
        "customer_message": "My app keeps crashing every time I open it on my iPhone.",
        "correct_action": "troubleshoot",
        "expected_keywords": ["restart", "update", "reinstall", "steps"],
        "max_steps": 3,
    },
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

# Hard limits — validator rejects exactly 0.0 and exactly 1.0
REWARD_MIN = 0.01
REWARD_MAX = 0.99


def clamp_reward(x: float) -> float:
    """Guarantee reward is strictly inside (0, 1). Always call this before returning."""
    clamped = max(REWARD_MIN, min(REWARD_MAX, float(x)))
    return round(clamped, 4)


def grade(task: dict, action_type: str, response: str) -> tuple[float, str]:
    """
    Score an agent action. Returns (reward, feedback).
    reward is ALWAYS strictly in (0.01, 0.99).

    Max possible raw score = 0.48 + 0.29 + 0.19 = 0.96 → clamped to 0.96
    Min possible raw score = 0.0 → clamped to 0.01
    """
    reward = 0.0
    notes = []

    response = response or ""
    response_lower = response.lower()

    # 1. Action correctness — max 0.48 (not 0.50, keeps ceiling below 1.0)
    if action_type == task["correct_action"]:
        reward += 0.48
        notes.append("Correct action type (+0.48)")
    else:
        notes.append(f"Wrong action '{action_type}', expected '{task['correct_action']}' (+0.00)")

    # 2. Keyword coverage — max 0.29
    keywords_hit = [kw for kw in task["expected_keywords"] if kw in response_lower]
    kw_ratio = len(keywords_hit) / max(len(task["expected_keywords"]), 1)
    kw_score = round(kw_ratio * 0.29, 4)
    reward += kw_score
    notes.append(f"Keywords {keywords_hit} +{kw_score:.2f}")

    # 3. Quality — max 0.19
    quality = 0.0
    words = len(response.split())
    if words >= 20:
        quality += 0.07
    if words >= 50:
        quality += 0.04
    polite = ["sorry", "apologize", "thank", "understand", "appreciate", "please"]
    if any(p in response_lower for p in polite):
        quality += 0.05
    bad = ["not my problem", "you're wrong", "don't care"]
    if not any(b in response_lower for b in bad):
        quality += 0.03
    quality = round(min(quality, 0.19), 4)
    reward += quality
    notes.append(f"Quality +{quality:.2f} (words={words})")

    # ALWAYS clamp — this is the only place reward is returned
    reward = clamp_reward(reward)
    return reward, " | ".join(notes)
