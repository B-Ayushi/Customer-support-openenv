"""
main.py — FastAPI server implementing the OpenEnv step/reset/state API
for the Customer Support Ticket Resolution environment.
"""

import random
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.models import Action, Observation, State
from server.tasks import TASKS, grade, clamp_reward

app = FastAPI(
    title="Customer Support RL Environment",
    description="OpenEnv-compliant environment for customer support ticket resolution.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/")
def health():
    return {"status": "ok", "environment": "customer-support-rl"}


@app.post("/reset", response_model=Observation)
def reset(task_id: str | None = None):
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
        reward=0.01,   # strictly > 0
        feedback="Episode started. Respond to the customer ticket.",
        session_id=session_id,
    )


@app.post("/step", response_model=Observation)
def step(session_id: str, action: Action):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    state = _sessions[session_id]

    if state["done"]:
        raise HTTPException(status_code=400, detail="Episode already done. Call /reset.")

    state["step_count"] += 1

    raw_reward, feedback = grade(
        task={
            "correct_action": state["correct_action"],
            "expected_keywords": state["expected_keywords"],
        },
        action_type=action.action_type,
        response=action.response,
    )

    # clamp_reward guarantees strictly (0.01, 0.99) — never 0.0 or 1.0
    reward = clamp_reward(raw_reward)

    state["total_reward"] += reward
    state["history"].append({
        "step": state["step_count"],
        "action_type": action.action_type,
        "response": action.response,
        "reward": reward,
    })

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
        session_id=session_id,
    )


@app.get("/state", response_model=State)
def get_state(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    s = _sessions[session_id]
    return State(**{k: v for k, v in s.items() if k != "history"})


@app.get("/tasks")
def list_tasks():
    return [
        {"ticket_id": t["ticket_id"], "difficulty": t["difficulty"], "customer_message": t["customer_message"]}
        for t in TASKS
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=7860, reload=False)
