"""
tasks.py — ticket bank and grader logic.
Clamped to (0.05, 0.85) to ensure total cumulative reward never hits 1.0.
"""

TASKS = [
    {
        "ticket_id": "T001",
        "difficulty": "easy",
        "customer_message": "Hi, I was charged twice for my subscription this month. I need my money back as soon as possible.",
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
        "customer_message": "I cancelled my plan three days ago but I was still charged today. I've already tried contacting billing twice and nobody helped me.",
        "correct_action": "refund",
        "expected_keywords": ["apologize", "refund", "billing", "investigate"],
        "max_steps": 4,
    },
    {
        "ticket_id": "T005",
        "difficulty": "medium",
        "customer_message": "I think there might be unauthorized access to my account. I received a login alert from a country I've never been to.",
        "correct_action": "escalate",
        "expected_keywords": ["security", "team", "escalate", "urgent", "lock"],
        "max_steps": 4,
    },
    {
        "ticket_id": "T006",
        "difficulty": "medium",
        "customer_message": "The dashboard is loading very slowly and some charts aren't showing. It started after your maintenance window yesterday.",
        "correct_action": "troubleshoot",
        "expected_keywords": ["cache", "browser", "refresh", "status", "issue"],
        "max_steps": 4,
    },
    {
        "ticket_id": "T007",
        "difficulty": "hard",
        "customer_message": "I run a small business and your API has been down for 6 hours. We've lost thousands of dollars. I need a refund AND I need to speak to someone senior.",
        "correct_action": "escalate",
        "expected_keywords": ["escalate", "senior", "apologize", "outage", "compensate"],
        "max_steps": 5,
    },
    {
        "ticket_id": "T008",
        "difficulty": "hard",
        "customer_message": "Your AI feature gave my client completely wrong financial advice. We used it in good faith and now we might face legal action.",
        "correct_action": "escalate",
        "expected_keywords": ["legal", "escalate", "logs", "team", "sorry", "policy"],
        "max_steps": 5,
    },
    {
        "ticket_id": "T009",
        "difficulty": "hard",
        "customer_message": "I'm a developer. I've followed all your docs but the OAuth2 flow keeps returning a 401. I need help debugging.",
        "correct_action": "troubleshoot",
        "expected_keywords": ["headers", "token", "cors", "redirect", "debug", "oauth"],
        "max_steps": 5,
    },
]

# SAFE BOUNDS: Never allow a 0.0 or 1.0 total
REWARD_MIN = 0.05
REWARD_MAX = 0.85

def clamp_reward(reward: float) -> float:
    """Strictly forces reward into the safe zone."""
    return float(max(REWARD_MIN, min(REWARD_MAX, reward)))

def grade(task: dict, action_type: str, response: str) -> tuple[float, str]:
    reward = 0.0
    notes = []
    response = response or ""
    res_lower = response.lower()

    # 1. Action Correctness (Max 0.45)
    if action_type == task["correct_action"]:
        reward += 0.45
        notes.append("Correct action type (+0.45)")
    else:
        notes.append(f"Expected {task['correct_action']} (+0.00)")

    # 2. Keyword Coverage (Max 0.30)
    expected_kws = task.get("expected_keywords", [])
    if expected_kws:
        hits = [kw for kw in expected_kws if kw in res_lower]
        kw_score = (len(hits) / len(expected_kws)) * 0.30
        reward += kw_score
        notes.append(f"Keywords {len(hits)}/{len(expected_kws)} (+{kw_score:.2f})")

    # 3. Quality (Max 0.15)
    quality = 0.0
    words = len(response.split())
    if words >= 20: quality += 0.05
    if any(p in res_lower for p in ["sorry", "apologize", "thank", "please"]):
        quality += 0.05
    if not any(b in res_lower for b in ["not my problem", "you're wrong"]):
        quality += 0.05
    
    reward += quality
    notes.append(f"Quality (+{quality:.2f})")

    # Final Safety Clamp
    final_reward = clamp_reward(reward)
    return final_reward, " | ".join(notes)