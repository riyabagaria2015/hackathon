"""
Day 3: Gradio multi-actor streaming chat with stateful sessions.

- `gr.State` holds `characters` plus `messages` (Chatbot history as role/content dicts).
- Three buttons start a *word-by-word* streaming assistant bubble for Alice, Bob, or Charlie.
- A shared stream \"epoch\" counter lets a new actor preempt an in-flight stream so Bob can
  visually cut off Alice without waiting for her script to finish (requires `queue()` with
  concurrency so another event can run while this generator yields).

Run: `uv run python src/day3_streaming_ui.py`

Important: any `gr.State` must be constructed **inside** `with gr.Blocks():` so Gradio
registers it; creating State outside the context causes `KeyError` on every button click.
"""

from __future__ import annotations

import copy
import functools
import threading
import time
from collections.abc import Iterator
from typing import Any

import gradio as gr

# --- Preemption (single-browser demo; global epoch is enough for the hackathon UI story) ---
_stream_epoch = [0]
_epoch_lock = threading.Lock()


def _bump_stream_epoch() -> int:
    with _epoch_lock:
        _stream_epoch[0] += 1
        return _stream_epoch[0]


def _current_epoch() -> int:
    return _stream_epoch[0]


# Word lists long enough to click another actor mid-stream (~0.12s per word).
SCRIPTS: dict[str, list[str]] = {
    "Alice": """
        I am opening the quarterly narrative and the first thing I want on the record is
        that our team was never late without a documented dependency we escalated three
        times in writing and nobody wants to hear that because it complicates the story
        leadership prefers about velocity and optics and I think that is unfair to us
    """.split(),
    "Bob": """
        Hold on hold on before we relitigate week one can we agree the blocker queue sat
        untouched while dashboards turned green and that green was synthetic because the
        integration tests were skipped and everyone knew it in the room but stayed quiet
    """.split(),
    "Charlie": """
        Point of order we are about to lose the panel if we spiral can we anchor on three
        numbered risks assign owners in sixty seconds and park personality conflicts for
        a breakout because the clock is real and the sponsor is watching this thread live
    """.split(),
}


def initial_session_state() -> dict[str, Any]:
    return {
        "characters": ["Alice", "Bob", "Charlie"],
        "messages": [],
    }


def stream_actor(
    actor: str,
    session: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], list[dict[str, str]]]]:
    """
    Generator: append one assistant bubble for `actor` and stream words into it.

    Yields `(updated_session, messages_for_chatbot)` so `gr.State` and `gr.Chatbot` stay in sync.
    If another actor's button bumps the epoch, we stop early and mark the bubble as cut off.
    """
    epoch = _bump_stream_epoch()
    session = copy.deepcopy(session)
    messages: list[dict[str, str]] = copy.deepcopy(session.get("messages", []))

    header = f"[{actor}] "
    messages.append({"role": "assistant", "content": header})
    words = SCRIPTS.get(actor, ["…"])

    interrupted = False
    for word in words:
        if _current_epoch() != epoch:
            interrupted = True
            break
        messages[-1]["content"] = messages[-1]["content"] + word + " "
        session["messages"] = copy.deepcopy(messages)
        yield session, messages
        time.sleep(0.12)

    if interrupted and messages[-1]["content"].strip() != header.strip():
        messages[-1]["content"] = messages[-1]["content"].rstrip() + " *[stream cut — another actor took the floor]*"
        session["messages"] = copy.deepcopy(messages)
        yield session, messages


def reset_scene(_session: dict[str, Any]) -> tuple[dict[str, Any], list, str]:
    """Clear chat, bump epoch so any sleeping stream stops, return empty bot."""
    _bump_stream_epoch()
    cleared = initial_session_state()
    return cleared, [], "Scene cleared. Trigger an actor to stream."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Day 3 — Multi-actor streaming") as demo:
        # Must be created INSIDE `with gr.Blocks()` — otherwise the State is not wired into
        # the Blocks config and every click raises KeyError in SessionState (block _id).
        session_state = gr.State(initial_session_state())

        gr.Markdown(
            "## Multi-bubble streaming scene\n"
            "Trigger **Alice**, **Bob**, or **Charlie** to stream a monologue into the chatbot. "
            "While one actor is streaming, click another — the new bubble should start and the "
            "previous stream should stop (epoch preemption + concurrent queue). "
            "Uses `gr.State` for session + `gr.Chatbot` (role/content messages)."
        )
        chatbot = gr.Chatbot(
            label="Scene transcript",
            height=420,
        )
        status = gr.Textbox(
            label="Hint",
            value="Trigger an actor. While they stream, click a different actor to cut in.",
            interactive=False,
        )

        with gr.Row():
            btn_alice = gr.Button("Trigger Alice", variant="primary")
            btn_bob = gr.Button("Trigger Bob", variant="primary")
            btn_charlie = gr.Button("Trigger Charlie", variant="primary")
        clear = gr.Button("Clear scene")

        # Concurrency lets a second button handler run while the first generator is sleeping between yields.
        demo.queue(default_concurrency_limit=20)

        # Bind actor name with partial — do NOT use extra `gr.State(actor)` inputs. Inline
        # `gr.State(...)` inside a loop is not wired into Gradio's session map and causes
        # KeyError on click (state_holder cannot resolve block _id for that State).
        for actor, button in (
            ("Alice", btn_alice),
            ("Bob", btn_bob),
            ("Charlie", btn_charlie),
        ):
            button.click(
                functools.partial(stream_actor, actor),
                inputs=[session_state],
                outputs=[session_state, chatbot],
            )

        clear.click(
            reset_scene,
            inputs=[session_state],
            outputs=[session_state, chatbot, status],
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
