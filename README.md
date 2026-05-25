# hackathon

## Day 1 — Fake Interruption Engine

Two concurrent “scenes” simulate streaming dialogue: **Actor A** adds one word per second to a scene-local buffer; **Listener B** polls every 2 seconds. When the buffer contains the trigger word **`apple`**, B sets an `asyncio.Event`, cancels A, and streams its own words. Each scene runs under `contextvars.copy_context()` so tenant state does not mix.

**Why a `BufferState` object?** `asyncio` copies context into each new `Task`, so rebinding a `ContextVar` to a new string in one task would not update a sibling task. The buffer is a **mutable** object stored in a `ContextVar` once per scene; co-tasks mutate it in place, while different scenes still get different instances thanks to `copy_context()`.

## Run

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv run python src/fake_interruption_engine.py
```

Optional template entrypoint:

```bash
uv run python main.py
```

Python **3.11+** is required (`requires-python` in `pyproject.toml`).
