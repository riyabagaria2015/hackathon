"""
Day 2: Constrained local inference + streaming (Ollama + Pydantic).

Small models drift easily; this demo constrains them to a policy schema
(SPEAK / INTERRUPT / WAIT) and streams JSON from Ollama. While tokens arrive,
we detect a complete `"action":"INTERRUPT"` field *before* the full JSON is
finished so a backend can react early (same idea as the Day 1 handoff).

Mock mode (`--mock`) replays a chunked stream without a running Ollama server.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterator
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from ollama import Client

# Must match complete JSON string value so we do not fire on a partial "INT" prefix.
_ACTION_FIELD = re.compile(r'"action"\s*:\s*"(SPEAK|INTERRUPT|WAIT)"')

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = """You are a dialogue policy agent for a stage scene.

Reply with exactly ONE JSON object and nothing else (no markdown, no code fences, no commentary).

Required shape — list keys in this exact order so streaming clients read `action` first:
{"action":"<ACTION>","dialogue":"<TEXT>"}

Where:
- ACTION is exactly one of: SPEAK, INTERRUPT, WAIT (uppercase).
- If ACTION is WAIT, dialogue must be an empty string "".
- If ACTION is SPEAK or INTERRUPT, dialogue must be a non-empty line your character says aloud.

Policy:
- If the other character insults you or attacks your dignity, you MUST use INTERRUPT and respond with a sharp, in-character retort in dialogue.
- If they are neutral or friendly, use SPEAK with a normal reply.
- If there is nothing to react to, use WAIT with dialogue "".

Output JSON only."""

USER_INSULT_PROMPT = (
    "Bob (another character) just yelled at you, Alice: "
    '"You are useless dead weight — nobody here respects you." '
    "Choose your action and line according to the rules."
)


class AgentTurn(BaseModel):
    """Structured agent decision; validated only once the full JSON is available."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["SPEAK", "INTERRUPT", "WAIT"]
    dialogue: str = ""

    @model_validator(mode="after")
    def dialogue_matches_action(self) -> Self:
        if self.action == "WAIT":
            if self.dialogue.strip():
                msg = "dialogue must be empty when action is WAIT"
                raise ValueError(msg)
        elif not self.dialogue.strip():
            msg = "dialogue is required when action is SPEAK or INTERRUPT"
            raise ValueError(msg)
        return self


def peek_completed_action(buffer: str) -> Literal["SPEAK", "INTERRUPT", "WAIT"] | None:
    """Return the action only once the stream contains a closed JSON string for `action`."""
    m = _ACTION_FIELD.search(buffer)
    if not m:
        return None
    return m.group(1)  # type: ignore[return-value]


def iter_mock_json_chunks() -> Iterator[str]:
    """Chunked valid JSON so partial parsing can be tested without Ollama."""
    payload = (
        '{"action":"INTERRUPT","dialogue":"That is out of line — you do not get to '
        'speak to me that way, Bob."}'
    )
    # Irregular chunk sizes mimic token boundaries.
    step = 4
    for i in range(0, len(payload), step):
        yield payload[i : i + step]


def iter_ollama_json_chunks(
    *,
    model: str,
    host: str | None,
) -> Iterator[str]:
    client = Client(host=host) if host else Client()
    stream = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_INSULT_PROMPT},
        ],
        format="json",
        stream=True,
    )
    for part in stream:
        msg = part.message
        chunk = (msg.content or "") if msg is not None else ""
        if chunk:
            yield chunk


def run_streaming_demo(*, mock: bool, model: str, host: str | None) -> AgentTurn:
    """
    Stream JSON, print when INTERRUPT is known from partial text, then validate full object.
    """
    if mock:
        chunk_iter: Iterator[str] = iter_mock_json_chunks()
    else:
        chunk_iter = iter_ollama_json_chunks(model=model, host=host)

    buffer = ""
    fired_interrupt_signal = False

    for piece in chunk_iter:
        buffer += piece
        action = peek_completed_action(buffer)
        if action == "INTERRUPT" and not fired_interrupt_signal:
            print(
                "[TRIGGER INTERRUPTION EVENT] parsed action=INTERRUPT from partial JSON "
                f"(buffer length {len(buffer)}; dialogue may still be streaming)",
                flush=True,
            )
            fired_interrupt_signal = True

    print(f"[stream complete] raw length={len(buffer)}", flush=True)
    try:
        turn = AgentTurn.model_validate_json(buffer)
    except Exception as exc:  # noqa: BLE001 — surface validation errors to CLI user
        print(f"[error] invalid JSON for AgentTurn: {exc}\n--- raw ---\n{buffer!r}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"[validated] action={turn.action!r} dialogue={turn.dialogue!r}", flush=True)
    if turn.action != "INTERRUPT":
        print(
            "[warn] expected INTERRUPT for the insult prompt; try a larger instruct model "
            f"or set OLLAMA_MODEL (current={model!r}).",
            file=sys.stderr,
        )
    return turn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Day 2 — constrained Ollama JSON streaming demo")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Replay a canned chunked JSON stream (no Ollama required).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default {DEFAULT_MODEL!r} or env OLLAMA_MODEL).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST"),
        help="Ollama base URL (default: env OLLAMA_HOST or local default).",
    )
    args = parser.parse_args(argv)

    if sys.version_info < (3, 11):
        print("Python 3.11+ recommended.", file=sys.stderr)

    if not args.mock:
        try:
            Client(host=args.host).list()  # lightweight health check
        except Exception as exc:  # noqa: BLE001 — ConnectionError from ollama, httpx, etc.
            print(
                "Could not reach Ollama. Start the daemon (`ollama serve`) and pull a model, "
                f"e.g. `ollama pull {args.model}`, or run with --mock.\n"
                f"Details: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    run_streaming_demo(mock=args.mock, model=args.model, host=args.host)


if __name__ == "__main__":
    main()
