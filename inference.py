"""
inference.py — Baseline inference script for the Customer Support RL Environment.

Output format (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Environment variables with defaults (required by hackathon rules) ──────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI client (mandatory per hackathon rules) ─────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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


def call_llm(customer_message: str, history: list) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({
            "role": "assistant",
            "content": json.dumps({"action_type": h["action_type"], "response": h["response"]})
        })
        messages.append({
            "role": "user",
            "content": f"Previous reward: {h['reward']:.2f}. Please improve your response."
        })
    messages.append({"role": "user", "content": f"Customer message:\n{customer_message}"})

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )
    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
       return json.loads(raw.strip())
    except:
       raw = raw.replace("\n", " ").replace("\t", " ")
       return json.loads(raw)


def run_episode(task_id: str) -> dict:
    reset_resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    session_id       = obs["session_id"]
    customer_message = obs["customer_message"]

    print(f"[START] task={task_id} env=customer-support model={MODEL_NAME}")

    step_count = 0
    rewards    = []
    done       = False
    success    = False
    last_error = None

    while not done:
        try:
            action = call_llm(customer_message, obs.get("history", []))
            last_error = None
        except Exception as e:
            last_error = str(e)
            action = {"action_type": "inform", "response": "I apologize, I encountered an error."}

        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                params={"session_id": session_id},
                json=action,
            )
            step_resp.raise_for_status()
            obs = step_resp.json()

            step_count += 1
            reward      = obs["reward"]
            done        = obs["done"]
            rewards.append(reward)

            if done and reward >= 0.8:
                success = True

            error_str = last_error if last_error else "null"
            print(
                f"[STEP] step={step_count} "
                f"action={action['action_type']} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={error_str}"
            )

        except Exception as e:
            step_count += 1
            rewards.append(0.0)
            done = True
            print(
                f"[STEP] step={step_count} "
                f"action={action.get('action_type', 'unknown')} "
                f"reward=0.00 done=true error={str(e)}"
            )

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={'true' if success else 'false'} steps={step_count} rewards={rewards_str}")

    return {
        "task_id":    task_id,
        "success":    success,
        "steps":      step_count,
        "avg_reward": round(sum(rewards) / max(len(rewards), 1), 4),
    }


def main():
    results = []
    for task_id in TASKS:
        try:
            result = run_episode(task_id)
            results.append(result)
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=0.00")
            results.append({"task_id": task_id, "success": False, "steps": 0, "avg_reward": 0.0})

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['task_id']}  avg_reward={r['avg_reward']:.4f}  steps={r['steps']}")
    overall = sum(r["avg_reward"] for r in results) / max(len(results), 1)
    print(f"\n  OVERALL avg_reward: {overall:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()