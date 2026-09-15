# Gym AI Customer Voice Agent — Multi-Agent Support Bot

A multi-agent customer-support system for a fitness facility, built with **LangChain / LangGraph** multi-agent **handoffs**, traced with **LangSmith**, and served as both a **text chat** and a real-time **Pipecat voice bot** from the same state graph.

It demonstrates the **"agents as graph nodes"** pattern: a **triage** front desk plus three specialists — **cancellation**, **credits**, and **booking** — each a distinct node in the graph. The customer stays "inside" a specialist across turns because the active agent is stored in state.

> The graph re-enters at `START` every turn and routes straight to the `active_agent`. State remembers where the conversation is — the engine does not freeze you inside a node.

```
START ──(route_initial: active_agent or "triage")──► triage ──┐
                                                               ├─► cancellation ─┐
                                                               ├─► credits ──────┼─► (route_after_agent) ─► END
                                                               └─► booking ──────┘            ▲
                                                   specialists ── transfer_to_triage ─────────┘
```

Each agent's handoff tools (`transfer_to_cancellation`, `transfer_to_credits`, `transfer_to_booking`, `transfer_to_triage`) return a `Command(goto=..., graph=Command.PARENT)` that jumps to a sibling node and updates `active_agent` using the LangChain / LangGraph multi-agent handoff pattern.

---

## 🌟 Key Capabilities

- **Cancel a membership:** Give any ID; it is verified and processed in state.
- **Check credits:** Returns a mock credit breakdown (group class / personal training / guest passes).
- **Book a class:** Lists a schedule and spends one credit per booking.
- **Real-Time Voice Bot:** Pipeline integrating Speech-to-Text (STT), Claude Sonnet LLM graph processing, and Text-to-Speech (TTS) with full barge-in and interruption handling via Pipecat.

All data is mocked in-process (`mock_data.py`). There is no external database dependency: tool side effects mutate an in-memory dictionary that cleanly resets on server restart.

---

## 📁 Repository Layout

```
src/gym_support/
├── graph.py                 # Core 4-agent nodes & handoff routing
├── tools.py                 # Business logic tools + transfer_to_* handoffs
├── prompts.py               # System prompts per agent
├── mock_data.py             # In-memory membership data store
├── server.py                # FastAPI text chat (GET / page + POST /chat)
├── voice.py                 # Pipecat real-time voice bot
├── langgraph_llm_service.py # Adapter running the graph as Pipecat LLM stage
├── eval.py                  # Evaluation suite measuring latency & routing
└── static/
    └── index.html           # Interactive web chat interface
```

---

## 🚀 Setup & Execution

### Prerequisites
Requires **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)**.

```bash
uv sync
cp .env.example .env
```

### Environment Variables (`.env`)
- `ANTHROPIC_API_KEY` — Required; agent model execution (`anthropic:claude-sonnet-4-6`).
- `OPENAI_API_KEY` — Required for the **voice bot** (Speech-to-Text and Text-to-Speech).
- `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` — Optional; enables full graph execution tracing in LangSmith.

---

## 💻 Running the Applications

### 1. Web Text Chat (FastAPI)
```bash
uv run python -m gym_support.server
```
Open `http://127.0.0.1:8000` to interact with the web chat interface. Try:
- *"I want to cancel my membership"* → Watch the badge route to **cancellation**.
- *"How many credits do I have left?"* → Routes to **credits**.
- *"I'd like to book a class"* → Routes to **booking**.

### 2. Real-Time Voice Bot (Pipecat)
```bash
uv run python -m gym_support.voice
```
Open `http://localhost:7860`, click **Connect**, allow microphone access, and speak directly to the AI agent.

### 3. Evaluation Suite
```bash
uv run python -m gym_support.eval
```
Evaluates latency, routing correctness across intent switches, and reply truthfulness against `mock_data`.

---

## 📜 License

This project is licensed under the MIT License.
