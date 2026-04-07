"""
main.py — FastAPI server implementing the OpenEnv step/reset/state API
for the Customer Support Ticket Resolution environment.
"""

import random
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.models import Action, Observation, State
from server.tasks import TASKS, grade

app = FastAPI(
    title="Customer Support RL Environment",
    description="OpenEnv-compliant environment for training agents on real-world customer support ticket resolution.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
# Maps session_id → current environment state dict
_sessions: dict[str, dict] = {}


def _new_state(task: dict) -> dict:
    return {
        "ticket_id": task["ticket_id"],
        "customer_message": task["customer_message"],
        "correct_action": task["correct_action"],
        "expected_keywords": task["expected_keywords"],
        "task_difficulty": task["difficulty"],
        "max_steps": task["max_steps"],
        "step_count": 0,
        "done": False,
        "total_reward": 0.0,
        "history": [],
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "environment": "customer-support-rl"}


@app.post("/reset", response_model=Observation)
def reset(task_id: str | None = None):
    """
    Start a new episode. Optionally pin a specific ticket via task_id.
    Returns the first observation.
    """
    session_id = str(uuid.uuid4())

    if task_id:
        task = next((t for t in TASKS if t["ticket_id"] == task_id), None)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    else:
        task = random.choice(TASKS)

    state = _new_state(task)
    _sessions[session_id] = state

    return Observation(
        ticket_id=state["ticket_id"],
        customer_message=state["customer_message"],
        history=[],
        task_difficulty=state["task_difficulty"],
        done=False,
        reward=0.0,
        feedback="Episode started. Respond to the customer ticket.",
        session_id=session_id,   # injected for client tracking
    )


@app.post("/step", response_model=Observation)
def step(session_id: str, action: Action):
    """
    Submit one agent action for the current episode.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    state = _sessions[session_id]

    if state["done"]:
        raise HTTPException(status_code=400, detail="Episode is already done. Call /reset to start a new one.")

    state["step_count"] += 1

    # Grade the action
    reward, feedback = grade(
        task={
            "correct_action": state["correct_action"],
            "expected_keywords": state["expected_keywords"],
        },
        action_type=action.action_type,
        response=action.response,
    )
    state["total_reward"] += reward
    state["history"].append(
        {"step": state["step_count"], "action_type": action.action_type, "response": action.response, "reward": reward}
    )

    # Episode ends if: max steps reached OR agent got a perfect score
    done = state["step_count"] >= state["max_steps"] or reward >= 0.95
    state["done"] = done

    return Observation(
        ticket_id=state["ticket_id"],
        customer_message=state["customer_message"],
        history=state["history"],
        task_difficulty=state["task_difficulty"],
        done=done,
        reward=reward,
        feedback=feedback,
    )


@app.get("/state", response_model=State)
def get_state(session_id: str):
    """Return the full internal state for a session (used by training harness)."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    s = _sessions[session_id]
    return State(**{k: v for k, v in s.items() if k != "history"})


@app.get("/tasks")
def list_tasks():
    """List all available tasks with their difficulties."""
    return [
        {"ticket_id": t["ticket_id"], "difficulty": t["difficulty"], "customer_message": t["customer_message"]}
        for t in TASKS
    ]
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=7860, reload=False)
