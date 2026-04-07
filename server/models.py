from pydantic import BaseModel
from typing import Optional, Literal, Any


# ── Action ────────────────────────────────────────────────────────────────────
class Action(BaseModel):
    """
    The agent submits a resolution response to a customer support ticket.
    Fields:
        response      : The natural-language reply the agent wants to send.
        action_type   : One of the four canonical resolution actions.
    """
    response: str
    action_type: Literal["refund", "escalate", "troubleshoot", "inform"]


# ── Observation ───────────────────────────────────────────────────────────────
class Observation(BaseModel):
    """
    What the agent sees after each step.
    Fields:
        ticket_id       : Unique identifier for the current ticket.
        customer_message: The raw complaint / query from the customer.
        history         : List of prior (action_type, response) pairs this episode.
        task_difficulty : 'easy' | 'medium' | 'hard'
        done            : Whether the episode has ended.
        reward          : Reward awarded for the last action (0.0 – 1.0).
        feedback        : Human-readable explanation of the reward.
    """
    ticket_id: str
    customer_message: str
    history: list[dict]
    task_difficulty: Literal["easy", "medium", "hard"]
    done: bool
    reward: float
    feedback: str
    session_id: Optional[str] = None   # returned by /reset for client tracking


# ── State ─────────────────────────────────────────────────────────────────────
class State(BaseModel):
    """Internal environment state (returned by /state endpoint)."""
    ticket_id: str
    customer_message: str
    correct_action: str
    expected_keywords: list[str]
    task_difficulty: str
    step_count: int
    max_steps: int
    done: bool
    total_reward: float
