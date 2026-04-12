import random
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.models import Action, Observation, State
from server.tasks import TASKS, grade, clamp_reward

app = FastAPI(title="Customer Support RL Environment")

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
        "total_reward": 0.12, # Safe start
        "history": [],
    }

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/reset", response_model=Observation)
def reset(task_id: str | None = None):
    session_id = str(uuid.uuid4())
    if task_id:
        task = next((t for t in TASKS if t["ticket_id"] == task_id), None)
        if not task: raise HTTPException(status_code=404)
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
        reward=0.12, # Strictly > 0 and room for growth
        feedback="Session started.",
        session_id=session_id,
    )

@app.post("/step", response_model=Observation)
def step(session_id: str, action: Action):
    if session_id not in _sessions:
        raise HTTPException(status_code=404)

    state = _sessions[session_id]
    if state["done"]:
        raise HTTPException(status_code=400, detail="Finished.")

    state["step_count"] += 1

    # Grade and clamp
    raw_reward, feedback = grade(
        task={"correct_action": state["correct_action"], "expected_keywords": state["expected_keywords"]},
        action_type=action.action_type,
        response=action.response,
    )
    
    reward = clamp_reward(raw_reward)
    state["total_reward"] = reward # Overwrite to prevent summing to > 1.0

    state["history"].append({
        "step": state["step_count"],
        "action_type": action.action_type,
        "reward": reward,
    })

    # End episode if good enough or too many steps
    done = state["step_count"] >= state["max_steps"] or reward >= 0.70
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
    if session_id not in _sessions: raise HTTPException(status_code=404)
    s = _sessions[session_id]
    return State(**{k: v for k, v in s.items() if k != "history"})
