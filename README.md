# hackathon

## Day 1 — Fake Interruption Engine

**Actor A** streams one word per second into a scene-local buffer; **Listener B** polls every 2 seconds. When the buffer contains **`apple`**, B sets an `asyncio.Event`, cancels A, and streams its own words. The demo runs a **single** scene for readable logs; multiple tenants would use `contextvars.copy_context()` per session so buffers do not mix.

**Why a `BufferState` object?** Each asyncio `Task` gets a context copy, so a `ContextVar` rebound to a new string in one task would not update siblings. The buffer is a **mutable** object stored once per scene; co-tasks mutate it in place.

```bash
uv run python src/fake_interruption_engine.py
```

## Day 2 — Constrained local inference & streaming

Small models drift; this demo constrains them to a **policy JSON object** validated with **Pydantic**:

- `action`: one of `SPEAK`, `INTERRUPT`, `WAIT`
- `dialogue`: required for `SPEAK` / `INTERRUPT`, must be `""` for `WAIT`

**Ollama** streams `format="json"` from a local model. While chunks arrive, a regex spots a complete `"action":"INTERRUPT"` field **before** the full JSON (and often before `dialogue` finishes), so a backend can emit `[TRIGGER INTERRUPTION EVENT]` early—same spirit as Day 1’s early handoff.

**Setup**

1. Install and start [Ollama](https://ollama.com/).
2. Pull a small instruct model, e.g. `ollama pull llama3.2` or `ollama pull phi3:mini`.
3. Optional: `export OLLAMA_MODEL=llama3.1:8b` (default is `llama3.2`).

```bash
uv run python src/day2_compliant_agent.py
```

**No Ollama?** Offline parser demo (canned chunked JSON):

```bash
uv run python src/day2_compliant_agent.py --mock
```

**Alternatives (not wired in this repo):** [instructor](https://github.com/jxnl/instructor) or [outlines](https://github.com/dottxt-ai/outlines) on top of Ollama for heavier schema enforcement; **llama-cpp-python** + **GBNF** grammars for strict JSON without a separate daemon—handy when you want token-level control inside one process.

## Other

```bash
uv run python main.py
```

Python **3.11+** is required (`requires-python` in `pyproject.toml`).
