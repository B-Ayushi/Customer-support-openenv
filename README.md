# 🎫 Customer Support Ticket Resolution — OpenEnv RL Environment

A real-world reinforcement learning environment where an AI agent learns to resolve customer support tickets by selecting the correct action type and crafting an appropriate response.

---

## 🌍 What does the agent learn?

Given a customer complaint, the agent must:
1. **Choose the correct action** from: `refund` | `escalate` | `troubleshoot` | `inform`
2. **Write a professional, empathetic reply** that includes the right resolution keywords

---

## 📊 Task Difficulty Tiers

| Tier   | Tickets          | Description |
|--------|-----------------|-------------|
| Easy   | T001, T002, T003 | Clear-cut, single correct action |
| Medium | T004, T005, T006 | Ambiguous, frustrated customers |
| Hard   | T007, T008, T009 | Multi-issue, business-critical, legal risk |

---

## 🏆 Reward Function (0.0 → 1.0)

| Component          | Weight | Description |
|--------------------|--------|-------------|
| Action correctness | 0.50   | Did the agent pick the right action type? |
| Keyword coverage   | 0.30   | Does the response include expected resolution words? |
| Response quality   | 0.20   | Length, politeness, no harmful phrases |

---

## 📁 Project Structure

```
openenv-support/
├── Dockerfile              ← Container definition (stays in root)
├── requirements.txt        ← Python dependencies
├── openenv.yaml            ← OpenEnv spec declaration
├── inference.py            ← Baseline inference script (MANDATORY in root)
├── .env.example            ← Copy to .env and fill in your tokens
└── server/
    ├── main.py             ← FastAPI app (reset / step / state)
    ├── models.py           ← Typed Pydantic models (Action, Observation, State)
    └── tasks.py            ← Ticket bank + grader logic
```

---

## 🚀 Local Setup (Step-by-Step for Beginners)

### Step 1 — Clone and enter the project
```bash
git clone <your-repo-url>
cd openenv-support
```

### Step 2 — Create your .env file
```bash
cp .env.example .env
# Open .env and paste your Hugging Face token
```
Get your free HF token at: https://huggingface.co/settings/tokens

### Step 3 — Run with Docker (recommended)
```bash
# Build the container
docker build -t customer-support-rl .

# Run it (maps container port 7860 to your machine's port 7860)
docker run -p 7860:7860 customer-support-rl
```

The server is now live at: http://localhost:7860

### Step 4 — Check it works
Open your browser and visit: http://localhost:7860/docs
You'll see the interactive API documentation.

### Step 5 — Run the inference script
```bash
# Install Python deps locally
pip install -r requirements.txt

# Run inference against your local server
python inference.py
```

---

## 🌐 Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Choose **Docker** as the SDK
3. Push this entire project to the Space repository
4. Add your `HF_TOKEN` as a Secret in the Space settings
5. The Space will auto-build and deploy

After deployment, update your `.env`:
```
ENV_URL=https://your-username-customer-support-rl.hf.space
```

---

## 🔌 API Reference

### `POST /reset?task_id=T001`
Start a new episode. Returns first observation.

### `POST /step?session_id=<id>`
Submit an action. Body:
```json
{
  "action_type": "refund",
  "response": "I'm sorry to hear about the double charge. I'll process a full refund immediately..."
}
```

### `GET /state?session_id=<id>`
Get full internal state (for training harnesses).

### `GET /tasks`
List all available tasks and difficulties.

---

## 📋 Pre-Submission Checklist

- [ ] `openenv.yaml` exists in root
- [ ] `Dockerfile` exists in root
- [ ] `inference.py` exists in root
- [ ] Server responds 200 to `GET /`
- [ ] `/reset` works and returns a valid observation
- [ ] `/step` accepts actions and returns rewards in [0.0, 1.0]
- [ ] All 9 tasks have graders returning valid reward scores
- [ ] `.env` has `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] Inference script completes in < 20 minutes
- [ ] Inference script prints `[START]`, `[STEP]`, `[END]` logs to stdout

---

## 💡 Tips

- The grader is **partial** — even a wrong action type can score up to 0.50 via keyword + quality signals
- Hard tasks allow up to 5 steps, so the agent can iterate and improve
- The `feedback` field in each observation tells the agent exactly what it got right/wrong
