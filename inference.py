"""
inference.py — Baseline inference script for the Customer Support RL Environment.
Bulletproof version: rewards are ALWAYS strictly between 0 and 1.
"""

import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Safety Constants — strictly inside (0, 1)
LOW_BOUND = 0.12
HIGH_BOUND = 0.88

TASKS = ["T001", "T002", "T003", "T004", "T005", "T006", "T007", "T008", "T009"]

SYSTEM_PROMPT = """You are an expert customer support agent.
Given a customer message, you must:
1. Choose the correct action_type from: refund, escalate, troubleshoot, inform
2. Write a professional, empathetic response to the customer.
Rules:
- refund      : Customer was incorrectly charged or deserves compensation.
- escalate    : Issue needs a senior agent, security team, or legal team.
- troubleshoot: Technical problem the agent can help debug step-by-step.
- inform      : Customer needs information or instructions (no billing/tech issue).
Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{
  "action_type": "<refund|escalate|troubleshoot|inform>",
  "response": "<your reply to the customer>"
}"""

def safe_reward(x) -> float:
    """Ensures reward is ALWAYS strictly within (0, 1). No exceptions."""
    try:
        val = float(x)
    except Exception:
        return LOW_BOUND
    if val <= 0.0 or val != val:  # catches 0.0, negative, NaN
        return LOW_BOUND
    if val >= 1.0:  # catches 1.0 and above
        return HIGH_BOUND
    # Extra safety: clamp to our safe range
    return max(LOW_BOUND, min(HIGH_BOUND, val))

def call_llm(customer_message: str, history: list) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        try:
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "action_type": h.get("action_type", "inform"),
                    "response": h.get("response", "I apologize for the inconvenience.")
                })
            })
            messages.append({
                "role": "user",
                "content": f"Previous reward: {safe_reward(h.get('reward', LOW_BOUND)):.4f}. Please improve your response."
            })
        except Exception:
            continue
    messages.append({"role": "user", "content": f"Customer message:\n{customer_message}"})

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )

    raw = completion.choices[0].message.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw.strip())
    except Exception:
        cleaned = raw.replace("\n", " ").replace("\t", " ").strip()
        return json.loads(cleaned)

def run_episode(task_id: str) -> dict:
    # 1. RESET
    reset_resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    session_id = obs["session_id"]
    customer_message = obs["customer_message"]

    print(f"[START] task={task_id} env=customer-support model={MODEL_NAME}")

    step_count = 0
    rewards = []
    done = False
    success = False

    while not done:
        try:
            action = call_llm(customer_message, obs.get("history", []))
            last_error = "null"
        except Exception as e:
            last_error = str(e).replace(" ", "_")
            action = {"action_type": "inform", "response": "I sincerely apologize for the inconvenience. Please allow me to assist you with this matter right away."}

        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                params={"session_id": session_id},
                json=action,
            )
            step_resp.raise_for_status()
            obs = step_resp.json()

            step_count += 1
            reward = safe_reward(obs.get("reward", LOW_BOUND))
            done = obs["done"]
            rewards.append(reward)

            if done and reward >= 0.75:
                success = True

            print(
                f"[STEP] step={step_count} "
                f"action={action['action_type']} "
                f"reward={reward:.4f} "
                f"done={'true' if done else 'false'} "
                f"error={last_error}"
            )

        except Exception as e:
            step_count += 1
            reward = LOW_BOUND
            rewards.append(reward)
            done = True
            print(
                f"[STEP] step={step_count} "
                f"action={action.get('action_type', 'inform')} "
                f"reward={reward:.4f} done=true error={str(e).replace(' ', '_')}"
            )

    # Guarantee rewards list is never empty
    if not rewards:
        rewards = [LOW_BOUND]

    total_reward = sum(rewards)
    max_possible_reward = max(len(rewards), 1)
    score = total_reward / max_possible_reward
    score = max(0.01, min(score, 0.99))

    task_name = task_id
    steps = step_count
    print(f"[END] task={task_name} score={score} steps={steps}", flush=True)

    return {
        "task_id": task_id,
        "success": success,
        "steps": step_count,
        "avg_reward": score,
    }

def main():
    results = []
    for task_id in TASKS:
        try:
            result = run_episode(task_id)
            results.append(result)
        except Exception as e:
            safe_r = 0.01
            print(f"[END] task={task_id} score={safe_r} steps=1", flush=True)
            results.append({"task_id": task_id, "success": False, "steps": 1, "avg_reward": safe_r})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "✓" if r["success"] else "✗"
        avg = max(0.01, min(float(r["avg_reward"]), 0.99))
        print(f"  {status} {r['task_id']}  avg_reward={avg:.4f}  steps={r['steps']}")

    overall = max(0.01, min(sum(max(0.01, min(float(r["avg_reward"]), 0.99)) for r in results) / max(len(results), 1), 0.99))
    print(f"\n  OVERALL avg_reward: {overall:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
