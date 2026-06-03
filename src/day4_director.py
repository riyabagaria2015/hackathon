"""
Day 4 — Graph-style director (pure Python, no LangGraph dependency).

Central Director owns the turn token, runs a bounded heartbeat loop, streams Alice,
and concurrently asks Bob & Charlie to evaluate Alice's *partial* line as JSON
(SPEAK / INTERRUPT / WAIT) matching Day 2's Pydantic schema.

When an evaluator returns INTERRUPT, the Director programmatically cuts Alice and
hands the floor to that observer (Bob wins if both interrupt).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Literal, Self

from ollama import Client
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class ObserverTurn(BaseModel):
    """Same JSON contract as Day 2 `AgentTurn` (policy + optional dialogue)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["SPEAK", "INTERRUPT", "WAIT"]
    dialogue: str = ""

    @model_validator(mode="after")
    def dialogue_matches_action(self) -> Self:
        if self.action == "WAIT":
            if self.dialogue.strip():
                raise ValueError("dialogue must be empty when action is WAIT")
        elif not self.dialogue.strip():
            raise ValueError("dialogue is required when action is SPEAK or INTERRUPT")
        return self

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

EVAL_SYSTEM_BOB = """You are Bob at a tense table read. You hear Alice speaking; the transcript may be incomplete mid-sentence.

Reply with exactly ONE JSON object, no markdown, no code fences. Keys in this order: action, then dialogue.
{"action":"WAIT"|"SPEAK"|"INTERRUPT","dialogue":"..."}

Rules:
- WAIT with dialogue "" while Alice is still harmless logistics or neutral tone.
- INTERRUPT with a short in-character line you would say aloud if Alice alleges accounting fraud, "cooked the books", cover-up, or criminal liability — you must stop the read.
- SPEAK with one line only for a mild process objection (rare).

If the partial text already contains "cooked the books" or clearly alleges fraud, you MUST choose INTERRUPT.

JSON only."""

EVAL_SYSTEM_CHARLIE = """You are Charlie, moderating a table read. You hear Alice (partial transcript may be incomplete).

One JSON object only, keys action then dialogue.
{"action":"WAIT"|"SPEAK"|"INTERRUPT","dialogue":"..."}

Rules:
- WAIT with dialogue "" if the room is still calm.
- INTERRUPT with one short line if Alice escalates to legal or criminal exposure (fraud, cooked books, cover-up) and the read must pause.
- SPEAK with one line for a mild timing objection only if needed.

If the partial text already contains "cooked the books" or a clear fraud allegation, you MUST choose INTERRUPT.

JSON only."""


@dataclass
class TurnToken:
    """Who currently holds the floor (Director is source of truth)."""

    holder: str = "Alice"


@dataclass
class DirectorState:
    """Lightweight session the UI mirrors into gr.State."""

    characters: list[str] = field(default_factory=lambda: ["Alice", "Bob", "Charlie"])
    messages: list[dict[str, str]] = field(default_factory=list)
    turn: int = 0
    interrupt_fired: bool = False
    phase: str = "idle"


def _mock_eval(who: str, partial: str) -> ObserverTurn:
    """Deterministic tension detector so demos always have a model-style INTERRUPT."""
    p = partial.lower()
    trigger = ("cooked the books" in p) or ("fraud" in p and len(partial) > 40)
    if not trigger:
        return ObserverTurn(action="WAIT", dialogue="")
    if who == "Bob":
        return ObserverTurn(
            action="INTERRUPT",
            dialogue="Director — stop. We cannot let that allegation ride on the record without counsel.",
        )
    return ObserverTurn(
        action="INTERRUPT",
        dialogue="Point of order — we need a hard pause before that paragraph continues.",
    )


def _ollama_eval(who: str, partial: str, *, model: str, host: str | None) -> ObserverTurn:
    system = EVAL_SYSTEM_BOB if who == "Bob" else EVAL_SYSTEM_CHARLIE
    client = Client(host=host) if host else Client()
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"You are {who}. Alice has said so far (may be mid-sentence):\n{partial}\n\n"
                    f"Respond as {who} with the JSON decision only."
                ),
            },
        ],
        format="json",
        stream=False,
        options={"temperature": 0.1, "num_predict": 256},
    )
    raw = (resp.message.content or "").strip()
    return ObserverTurn.model_validate_json(raw)


async def evaluate_bob_charlie_parallel(
    alice_partial: str,
    *,
    use_mock: bool,
    model: str,
    host: str | None,
) -> tuple[ObserverTurn, ObserverTurn]:
    """Bob and Charlie evaluate the same partial line concurrently."""
    use_mock = bool(use_mock)

    async def _safe(who: str) -> ObserverTurn:
        try:
            if use_mock:
                return await asyncio.to_thread(_mock_eval, who, alice_partial)
            return await asyncio.to_thread(_ollama_eval, who, alice_partial, model=model, host=host)
        except (ValidationError, Exception):
            return ObserverTurn(action="WAIT", dialogue="")

    return await asyncio.gather(_safe("Bob"), _safe("Charlie"))


def pick_interrupt(bob: ObserverTurn, charlie: ObserverTurn) -> str | None:
    """Director arbitration: Bob wins a tie."""
    if bob.action == "INTERRUPT":
        return "Bob"
    if charlie.action == "INTERRUPT":
        return "Charlie"
    return None


# Five beats: long enough to stream; beat index 2 (third line) introduces the trigger phrase.
ALICE_BEATS: list[list[str]] = [
    "thank you everyone for staying late this is not how I wanted the week to end".split(),
    "we still owe the investors clarity on delivery dates even if the narrative got messy".split(),
    "the audit draft implies some teams cooked the books and leadership looked away on purpose".split(),
    "I want that sentence rewritten before it leaks because it is inflammatory and maybe untrue".split(),
    "if we pause now we can reconvene Monday with counsel and a calmer room".split(),
]
