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

## Day 3 — Gradio streaming UI & stateful sessions

**`gr.State`** holds the list of **active characters** plus the **chat message list** fed to **`gr.Chatbot`** (each item is `{"role": "...", "content": "..."}`). Three buttons (**Trigger Alice / Bob / Charlie**) each run a **generator** that **yields** after every word so the UI streams.

**Cutting in mid-stream:** a shared **stream epoch** counter is bumped whenever a new actor starts; the running generator checks it on every iteration. **`demo.queue(default_concurrency_limit=20)`** lets another button’s handler start while the first stream is between yields (so the UI does not have to wait for Alice to finish). Stopped bubbles append a short *“stream cut”* suffix.

### How to use the UI

1. **Install & run** (from the repo root):

   ```bash
   uv run python src/day3_streaming_ui.py
   ```

2. **Open the app** in your browser: [http://127.0.0.1:7860](http://127.0.0.1:7860)  
   If that port is busy, stop the other process or change `server_port` in `src/day3_streaming_ui.py`.

3. **Trigger an actor** — click **Trigger Alice**, **Trigger Bob**, or **Trigger Charlie**.  
   A new assistant bubble appears and **streams word by word** in the chat transcript.

4. **Interrupt / hand off** — while one actor is still streaming, click **another** actor.  
   The first stream should stop (with a *stream cut* note) and the new actor’s bubble should start.

5. **Clear** — **Clear scene** wipes the chat and stops any in-flight stream.

**If buttons error after editing the code:** stop the server (**Ctrl+C**), save the file, and run the `uv run` command again so Gradio reloads the layout.

**Implementation note:** `gr.State` must be created **inside** `with gr.Blocks():`. If it is created outside, Gradio raises `KeyError` on clicks when resolving session state.

## Day 4 — Graph-based Director (table read)

A **Director** runs a **5-beat** “heartbeat” loop: each beat cues **Alice** to stream a line while **Bob** and **Charlie** (in parallel) evaluate her **partial** transcript as JSON (`ObserverTurn`: same contract as Day 2). If either returns **INTERRUPT**, the Director **programmatically** cuts Alice and streams the winner’s line (Bob wins ties). Rendered live in **Gradio** on **port 7861** (Day 3 can keep 7860).

### How to run Day 4

```bash
uv run python src/day4_table_read.py
```

Open **http://127.0.0.1:7861**.

1. **Default: real LLM** — leave **Mock evaluators** **unchecked**. Ensure **`ollama serve`** is running and run **`ollama pull <model>`** for the model in the textbox (defaults to `OLLAMA_MODEL` or `llama3.2`). Optional: set **`OLLAMA_HOST`** if the daemon is not on localhost.
2. Click **Start 5-turn table read** and watch the chat stream. On **beat 3**, Bob and Charlie are polled **every Alice word** so they see “cooked the books” in context; when either returns **INTERRUPT**, the Director cuts Alice and streams the winner’s line.
3. **Mock evaluators** — check the box only for a fully offline run (deterministic INTERRUPT on beat 3, no network).
4. **Reset** clears the transcript and session counters.

If Ollama is unreachable, the UI shows an error status and stops instead of silently falling back to mock.

## Day 5 — AI Puppet Theater (Gradio)

Stateful **AI Puppet Theater With Tiny Actors**: Backstage setup, curtain + chime, then **Play** with **Ollama**-generated dialogue (transcript-aware turns) or offline lines. Main Stage has **Use Ollama** + model name; interruptions only after several clauses so Actor 3 gets the floor. Wire HF TTS via `attach_hf_clients()` / `tts_synthesize`.

```bash
ollama serve   # elsewhere: ollama pull llama3.2
uv run python src/day5_puppet_theater.py
```

Open **http://127.0.0.1:7862**. **Spoken clauses** use [edge-tts](https://github.com/rany2/edge-tts) (neural MP3, needs internet); optional `EDGE_TTS_VOICE`, `EDGE_TTS_VOICE_DEEP`, `EDGE_TTS_VOICE_NERVOUS`, `EDGE_TTS_VOICE_ROBOT`. Also: `OLLAMA_HOST`, `OLLAMA_MODEL`, and `pip install huggingface_hub` for `InferenceClient`.

## Other

```bash
uv run python main.py
```

Python **3.11+** is required (`requires-python` in `pyproject.toml`).
