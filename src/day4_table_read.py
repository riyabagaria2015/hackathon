"""
Day 4 — Table read UI: Director heartbeat + live Gradio transcript.

Combines Day 1–3 ideas: asyncio-style concurrency for evaluators (via asyncio.run),
Day 2 JSON policy (`ObserverTurn`, same shape as `AgentTurn`), Day 3 Gradio streaming chat.

Run from repo root:
    uv run python src/day4_table_read.py

Uses port 7861 by default so Day 3 can stay on 7860.
"""

from __future__ import annotations

import asyncio
import copy
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import day4_director as d4  # noqa: E402
from ollama import Client  # noqa: E402


def _ollama_reachable(host: str | None) -> bool:
    try:
        Client(host=host).list()
        return True
    except Exception:
        return False


def _state_to_session(s: dict[str, Any]) -> d4.DirectorState:
    return d4.DirectorState(
        characters=list(s.get("characters", ["Alice", "Bob", "Charlie"])),
        messages=list(s.get("messages", [])),
        turn=int(s.get("turn", 0)),
        interrupt_fired=bool(s.get("interrupt_fired", False)),
        phase=str(s.get("phase", "idle")),
    )


def _session_to_state(sess: d4.DirectorState) -> dict[str, Any]:
    return {
        "characters": sess.characters,
        "messages": sess.messages,
        "turn": sess.turn,
        "interrupt_fired": sess.interrupt_fired,
        "phase": sess.phase,
    }


def _append(messages: list[dict[str, str]], role: str, content: str) -> None:
    messages.append({"role": role, "content": content})


def run_table_read(
    session: dict[str, Any],
    use_mock: bool,
    model: str,
) -> Iterator[tuple[dict[str, Any], list[dict[str, str]], str]]:
    """
    Five-turn director heartbeat, streamed into the Chatbot.

    Yields (session_dict, messages, status_line) for Gradio.
    """
    use_mock_b = bool(use_mock)
    model = (model or "").strip() or d4.DEFAULT_MODEL

    sess = _state_to_session(session)
    messages = copy.deepcopy(sess.messages)
    token = d4.TurnToken(holder="Alice")
    host = os.environ.get("OLLAMA_HOST")

    status = "Starting 5-turn table read (Director heartbeat)…"

    def emit() -> tuple[dict[str, Any], list[dict[str, str]], str]:
        sess.messages = copy.deepcopy(messages)
        return _session_to_state(sess), copy.deepcopy(messages), status

    yield emit()

    if not use_mock_b:
        if not _ollama_reachable(host):
            status = (
                "Ollama is not reachable. Start `ollama serve`, pull your model "
                f"(`ollama pull {model}`), or enable **Mock evaluators** for an offline run."
            )
            yield emit()
            return

    for turn in range(1, 6):
        sess.turn = turn
        sess.phase = "director_cue"
        token.holder = "Alice"
        status = f"Turn {turn}/5 — token={token.holder} — phase={sess.phase}"
        _append(
            messages,
            "assistant",
            f"[Director] Beat {turn}/5 — floor is **{token.holder}**. Bob & Charlie are evaluating partials.",
        )
        yield emit()

        sess.phase = "alice_stream"
        alice_line = d4.ALICE_BEATS[turn - 1]
        prefix = "[Alice] "
        _append(messages, "assistant", prefix)
        alice_slot = len(messages) - 1  # stable — do not use messages[-1] after appending eval rows
        yield emit()

        interrupted = False
        winner: str | None = None
        bob_d: d4.ObserverTurn | None = None
        charlie_d: d4.ObserverTurn | None = None

        for wi, word in enumerate(alice_line):
            if token.holder != "Alice":
                break
            messages[alice_slot]["content"] = messages[alice_slot]["content"] + word + " "
            status = f"Turn {turn}/5 — Alice streaming ({wi + 1}/{len(alice_line)} words)…"
            yield emit()
            time.sleep(0.08)

            # Mock: every word (cheap). LLM: every word on beat 3 so "cooked the books" is not skipped
            # between sparse polls; other beats poll every 2 words + final to save calls.
            if use_mock_b:
                should_eval = True
            elif turn == 3:
                should_eval = True
            else:
                should_eval = (wi % 2 == 1) or (wi == len(alice_line) - 1)
            if should_eval:
                partial = messages[alice_slot]["content"]
                sess.phase = "bc_eval"
                bob_d, charlie_d = asyncio.run(
                    d4.evaluate_bob_charlie_parallel(
                        partial,
                        use_mock=use_mock_b,
                        model=model,
                        host=host,
                    )
                )
                status = (
                    f"Turn {turn}/5 — evaluators: Bob={bob_d.action} Charlie={charlie_d.action} "
                    f'(token still "{token.holder}")'
                )
                _append(
                    messages,
                    "assistant",
                    f"[Director·eval] Bob → {bob_d.action} | Charlie → {charlie_d.action}",
                )
                yield emit()

                winner = d4.pick_interrupt(bob_d, charlie_d)
                if winner:
                    interrupted = True
                    sess.interrupt_fired = True
                    sess.phase = "interrupt"
                    token.holder = winner
                    if messages[alice_slot]["content"].startswith("[Alice]"):
                        messages[alice_slot] = dict(messages[alice_slot])
                        messages[alice_slot]["content"] = (
                            messages[alice_slot]["content"].rstrip()
                            + " *[Director: programmatic cut — "
                            + winner
                            + " INTERRUPT]*"
                        )
                    status = f"Turn {turn}/5 — INTERRUPT by {winner} (model-evaluated)"
                    _append(
                        messages,
                        "assistant",
                        f"[Director] Floor token → **{winner}**. Streaming interrupt line.",
                    )
                    yield emit()

                    line = (
                        bob_d.dialogue
                        if winner == "Bob"
                        else (charlie_d.dialogue if charlie_d else "")
                    )
                    if not line.strip():
                        line = "Stop the read — that crosses the line."
                    tag = f"[{winner}] "
                    _append(messages, "assistant", tag)
                    reply_slot = len(messages) - 1
                    for w in line.split():
                        messages[reply_slot]["content"] = messages[reply_slot]["content"] + w + " "
                        yield emit()
                        time.sleep(0.06)
                    token.holder = "Alice"
                    break

        if not interrupted:
            sess.phase = "beat_complete"
            status = f"Turn {turn}/5 — completed without interrupt"
            yield emit()

        # Prevent infinite mutual interrupts: after handling one, evaluators only WAIT on same beat tail
        # (we already broke inner loop). Director advances to next beat.

    sess.phase = "complete"
    status = "Table read complete (5 beats). Token returned to Alice for next rehearsal."
    _append(messages, "assistant", "[Director] End of automated 5-beat run.")
    yield emit()


def initial_session() -> dict[str, Any]:
    return _session_to_state(d4.DirectorState())


def reset_session(_: dict[str, Any]) -> tuple[dict[str, Any], list, str]:
    return initial_session(), [], "Ready. Click **Start 5-turn table read**."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Day 4 — Director table read") as demo:
        session_state = gr.State(initial_session())

        gr.Markdown(
            "## Day 4 — Graph-style Director (table read)\n"
            "A **Director** runs five beats. **Alice** streams each beat; **Bob** and **Charlie** "
            "evaluate her *partial* line in parallel via **Ollama** (`format=\"json\"`, same policy shape as Day 2). "
            "If either returns **INTERRUPT**, the Director **programmatically** cuts Alice and hands the floor "
            "to the winner (Bob wins ties). Check **Mock evaluators** only for offline demos without Ollama."
        )

        chatbot = gr.Chatbot(label="Live table read", height=460)
        status = gr.Textbox(label="Director status", value="Ready.", interactive=False)
        mock = gr.Checkbox(
            label="Mock evaluators (offline — deterministic INTERRUPT on beat 3)",
            value=False,
        )
        model = gr.Textbox(
            label="Ollama model",
            value=d4.DEFAULT_MODEL,
            info="Requires `ollama serve` and `ollama pull <model>`. Uses OLLAMA_HOST if set.",
        )

        with gr.Row():
            start = gr.Button("Start 5-turn table read", variant="primary")
            clear = gr.Button("Reset")

        demo.queue(default_concurrency_limit=5)

        start.click(
            run_table_read,
            inputs=[session_state, mock, model],
            outputs=[session_state, chatbot, status],
        )
        clear.click(reset_session, inputs=[session_state], outputs=[session_state, chatbot, status])

    return demo


def main() -> None:
    build_app().launch(server_name="127.0.0.1", server_port=7861, show_error=True)


if __name__ == "__main__":
    main()
