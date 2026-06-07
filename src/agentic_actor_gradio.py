"""Concept 14: agentic actor decisions in a Gradio puppet theater.

Run:
    uv run python src/agentic_actor_gradio.py

This demo makes "agentic" behavior visible:

- each actor has a goal, secret, memory, and tools
- actors produce structured decisions: WAIT, INTERRUPT, USE_TOOL
- the Director arbitrates requests and owns the turn
- tool results update scene memory
- audience events can push actors toward tool use or interruption
"""

from __future__ import annotations

import copy
import html
import time
from collections.abc import Iterator
from typing import Any, Literal, Self

import gradio as gr
from pydantic import BaseModel, ConfigDict, Field, model_validator

ActionKind = Literal["WAIT", "SPEAK", "INTERRUPT", "USE_TOOL"]


class ToolRequest(BaseModel):
    name: Literal["ledger_lookup", "memory_lookup", "prop_oracle"]
    query: str


class ActorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    actor: str
    action: ActionKind
    intent: str
    score: int = Field(ge=0, le=100)
    reason: str
    dialogue: str = ""
    tool_request: ToolRequest | None = None

    @model_validator(mode="after")
    def fields_match_action(self) -> Self:
        if self.action in {"SPEAK", "INTERRUPT"} and not self.dialogue:
            raise ValueError(f"{self.action} requires dialogue")
        if self.action == "USE_TOOL" and self.tool_request is None:
            raise ValueError("USE_TOOL requires tool_request")
        if self.action != "USE_TOOL" and self.tool_request is not None:
            raise ValueError("only USE_TOOL can include tool_request")
        return self


def default_actors() -> list[dict[str, Any]]:
    return [
        {
            "name": "Sock Dragon",
            "avatar": "🧦🐉",
            "goal": "Convince everyone he is a real dragon",
            "secret": "Actually knows he is a sock",
            "style": "grand, dramatic, oddly sincere",
            "tools": ["prop_oracle"],
            "memory": [],
            "cooldown": 0,
        },
        {
            "name": "Sir Croaksalot",
            "avatar": "🐸👑",
            "goal": "Become king without anyone inspecting the crown",
            "secret": "Already stole the crown",
            "style": "nervous knight, overreacts to audits",
            "tools": ["memory_lookup"],
            "memory": [],
            "cooldown": 0,
        },
        {
            "name": "Cactus Accountant",
            "avatar": "🌵📒",
            "goal": "Balance the kingdom budget",
            "secret": "Writes poetry in ledger margins",
            "style": "dry, precise, accidentally emotional",
            "tools": ["ledger_lookup"],
            "memory": [],
            "cooldown": 0,
        },
    ]


def initial_session() -> dict[str, Any]:
    return {
        "actors": default_actors(),
        "current_index": 0,
        "beat": 0,
        "audience_queue": [],
        "reality_rules": [],
        "transcript": [],
        "tool_log": [],
        "director_log": ["[Director] Session ready."],
        "last_speaker": None,
    }


def copy_session(session: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(session)


def actor_names(session: dict[str, Any]) -> list[str]:
    return [actor["name"] for actor in session["actors"]]


def current_speaker(session: dict[str, Any]) -> dict[str, Any]:
    actors = session["actors"]
    return actors[int(session["current_index"]) % len(actors)]


def listener_actors(session: dict[str, Any], speaker: dict[str, Any]) -> list[dict[str, Any]]:
    return [actor for actor in session["actors"] if actor["name"] != speaker["name"]]


def append_director(session: dict[str, Any], message: str) -> None:
    session.setdefault("director_log", []).append(message)
    session["director_log"] = session["director_log"][-80:]


def append_transcript(session: dict[str, Any], who: str, text: str) -> None:
    session.setdefault("transcript", []).append({"who": who, "text": text})
    session["transcript"] = session["transcript"][-80:]
    for actor in session["actors"]:
        actor.setdefault("memory", []).append(f"{who}: {text}")
        actor["memory"] = actor["memory"][-4:]


def apply_audience_queue(session: dict[str, Any]) -> None:
    while session["audience_queue"]:
        event = session["audience_queue"].pop(0)
        kind = event["kind"]
        payload = event["payload"]
        if kind == "audit":
            session["reality_rules"].append("An audit has been requested by the audience.")
            append_director(session, "[Audience] Audit request enters the scene.")
        elif kind == "prop":
            session["reality_rules"].append(f"A suspicious prop appears: {payload}.")
            append_director(session, f"[Audience] Throws prop: {payload}.")
        elif kind == "summon":
            new_actor = {
                "name": payload or "Oracle Spoon",
                "avatar": "🥄",
                "goal": "Make prophecies that complicate the current scene",
                "secret": "Can hear stage directions before they happen",
                "style": "paranoid, prophetic, theatrical",
                "tools": ["prop_oracle", "memory_lookup"],
                "memory": [],
                "cooldown": 0,
            }
            session["actors"].append(new_actor)
            append_director(session, f"[Audience] Summons new actor: {new_actor['name']}.")


def ledger_lookup(query: str) -> str:
    _ = query
    return "Royal Ledger: crown maintenance fund is missing 300 gold buttons."


def memory_lookup(session: dict[str, Any], actor_name: str, query: str) -> str:
    _ = query
    for actor in session["actors"]:
        if actor["name"] == actor_name:
            memories = actor.get("memory", [])
            return " | ".join(memories[-3:]) or "No useful memory yet."
    return "Actor memory not found."


def prop_oracle(query: str) -> str:
    if "duck" in query.lower():
        return "Prop Oracle: the rubber duck squeaks only when someone lies."
    return f"Prop Oracle: {query} hums with suspicious stage magic."


def run_tool(session: dict[str, Any], actor: dict[str, Any], request: ToolRequest) -> str:
    if request.name == "ledger_lookup":
        result = ledger_lookup(request.query)
    elif request.name == "memory_lookup":
        result = memory_lookup(session, actor["name"], request.query)
    else:
        result = prop_oracle(request.query)

    line = f"[Tool:{request.name}] {actor['name']} queried '{request.query}' → {result}"
    session.setdefault("tool_log", []).append(line)
    session["tool_log"] = session["tool_log"][-40:]
    append_director(session, line)
    return result


def speaker_line(session: dict[str, Any], speaker: dict[str, Any]) -> str:
    rules = "; ".join(session.get("reality_rules", [])[-2:]) or "normal stage rules"
    return (
        f"In this scene, my goal is to {speaker['goal']}. "
        f"I will proceed under {rules}."
    )


def decide_actor_action(
    session: dict[str, Any],
    actor: dict[str, Any],
    speaker: dict[str, Any],
    partial_text: str,
) -> ActorDecision:
    """Deterministic fake policy with the same shape we want from an LLM."""
    actor_name = actor["name"]
    lower = partial_text.lower()
    rules = " ".join(session.get("reality_rules", [])).lower()
    last_tool = " ".join(session.get("tool_log", [])[-2:]).lower()

    if actor_name == "Cactus Accountant" and "ledger_lookup" in actor["tools"]:
        if "audit" in rules and "ledger_lookup" not in last_tool:
            return ActorDecision(
                actor=actor_name,
                action="USE_TOOL",
                intent="investigate_budget",
                score=88,
                reason="audience requested an audit and this actor has ledger access",
                tool_request=ToolRequest(name="ledger_lookup", query="crown maintenance fund"),
            )

    if actor_name == "Sir Croaksalot":
        danger = "crown maintenance fund is missing" in last_tool or "crown" in lower
        if danger and int(actor.get("cooldown", 0)) == 0:
            return ActorDecision(
                actor=actor_name,
                action="INTERRUPT",
                intent="protect_secret",
                score=94,
                reason="the conversation threatens the stolen crown secret",
                dialogue="Objection! This audit is procedurally amphibious and deeply improper.",
            )

    if actor_name == "Sock Dragon" and "sock" in lower:
        return ActorDecision(
            actor=actor_name,
            action="INTERRUPT",
            intent="defend_identity",
            score=89,
            reason="speaker threatened Sock Dragon's identity",
            dialogue="I refuse this textile slander. I am scale, smoke, and destiny.",
        )

    return ActorDecision(
        actor=actor_name,
        action="WAIT",
        intent="observe",
        score=15,
        reason=f"{speaker['name']} has not crossed {actor_name}'s boundary",
    )


def pick_best_interrupt(decisions: list[ActorDecision]) -> ActorDecision | None:
    interrupts = [d for d in decisions if d.action == "INTERRUPT" and d.score >= 80]
    if not interrupts:
        return None
    return max(interrupts, key=lambda d: d.score)


def render_stage(session: dict[str, Any], active: str | None = None) -> str:
    cards = []
    for actor in session["actors"]:
        name = html.escape(actor["name"])
        avatar = html.escape(actor["avatar"])
        goal = html.escape(actor["goal"])
        tools = ", ".join(actor.get("tools", [])) or "none"
        cls = " actor-active" if actor["name"] == active else ""
        cards.append(
            f"""
<div class="actor-card{cls}">
  <div class="avatar">{avatar}</div>
  <strong>{name}</strong>
  <small>Goal: {goal}</small>
  <small>Tools: {html.escape(tools)}</small>
</div>
"""
        )
    return '<div class="stage">' + "".join(cards) + "</div>"


def transcript_chat(session: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": f"**{item['who']}:** {item['text']}"}
        for item in session.get("transcript", [])
    ]


def logs_text(session: dict[str, Any]) -> str:
    return "\n".join(session.get("director_log", [])[-30:])


def tool_text(session: dict[str, Any]) -> str:
    return "\n".join(session.get("tool_log", [])[-20:]) or "No tools used yet."


def ui_payload(session: dict[str, Any], active: str | None = None) -> tuple[Any, ...]:
    return (
        session,
        render_stage(session, active),
        transcript_chat(session),
        logs_text(session),
        tool_text(session),
    )


def run_one_beat(session: dict[str, Any]) -> Iterator[tuple[Any, ...]]:
    session = copy_session(session)
    apply_audience_queue(session)
    session["beat"] = int(session.get("beat", 0)) + 1

    speaker = current_speaker(session)
    listeners = listener_actors(session, speaker)
    append_director(
        session,
        f"[Director] Beat {session['beat']}: floor={speaker['name']}; "
        f"listeners={', '.join(a['name'] for a in listeners)}.",
    )
    yield ui_payload(session, speaker["name"])

    line = speaker_line(session, speaker)
    partial = ""
    for word in line.split():
        partial = f"{partial} {word}".strip()
        append_transcript(session, speaker["name"], partial)
        # Keep the transcript compact while streaming: replace the previous partial.
        if len(session["transcript"]) >= 2:
            previous = session["transcript"][-2]
            if previous["who"] == speaker["name"] and partial.startswith(previous["text"]):
                session["transcript"].pop(-2)
        yield ui_payload(session, speaker["name"])
        time.sleep(0.08)

    decisions = [decide_actor_action(session, actor, speaker, partial) for actor in listeners]
    for decision in decisions:
        append_director(
            session,
            f"[{decision.actor} policy] {decision.action} "
            f"score={decision.score} intent={decision.intent}; reason={decision.reason}",
        )
    yield ui_payload(session, speaker["name"])

    tool_decisions = [d for d in decisions if d.action == "USE_TOOL" and d.score >= 70]
    for decision in tool_decisions[:1]:
        actor = next(a for a in session["actors"] if a["name"] == decision.actor)
        assert decision.tool_request is not None
        result = run_tool(session, actor, decision.tool_request)
        append_transcript(session, "TOOL", result)
        yield ui_payload(session, decision.actor)

        decisions = [
            decide_actor_action(session, actor, speaker, result)
            for actor in listener_actors(session, speaker)
        ]
        for followup in decisions:
            append_director(
                session,
                f"[{followup.actor} follow-up] {followup.action} "
                f"score={followup.score}; reason={followup.reason}",
            )
        yield ui_payload(session, decision.actor)

    interrupt = pick_best_interrupt(decisions)
    if interrupt is not None:
        append_director(session, f"[Director] Interrupt approved for {interrupt.actor}.")
        actor = next(a for a in session["actors"] if a["name"] == interrupt.actor)
        actor["cooldown"] = 1
        append_transcript(session, interrupt.actor, interrupt.dialogue)
        session["current_index"] = session["actors"].index(actor)
        yield ui_payload(session, interrupt.actor)
    else:
        append_director(session, "[Director] No interrupt approved; normal turn advance.")

    for actor in session["actors"]:
        actor["cooldown"] = max(0, int(actor.get("cooldown", 0)) - 1)
    session["current_index"] = (int(session["current_index"]) + 1) % len(session["actors"])
    yield ui_payload(session, None)


def queue_audit(session: dict[str, Any]) -> tuple[Any, ...]:
    session = copy_session(session)
    session["audience_queue"].append({"kind": "audit", "payload": "crown budget"})
    append_director(session, "[Audience] Queued audit request.")
    return ui_payload(session)


def queue_prop(session: dict[str, Any], prop: str) -> tuple[Any, ...]:
    session = copy_session(session)
    session["audience_queue"].append({"kind": "prop", "payload": prop or "rubber duck"})
    append_director(session, f"[Audience] Queued prop: {prop or 'rubber duck'}.")
    return ui_payload(session)


def summon_actor(session: dict[str, Any], name: str) -> tuple[Any, ...]:
    session = copy_session(session)
    session["audience_queue"].append({"kind": "summon", "payload": name.strip() or "Oracle Spoon"})
    append_director(session, f"[Audience] Queued summon: {name.strip() or 'Oracle Spoon'}.")
    return ui_payload(session)


def reset_session() -> tuple[Any, ...]:
    return ui_payload(initial_session())


CSS = """
.stage {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.actor-card {
  border: 1px solid #3f3f46;
  border-radius: 8px;
  padding: 12px;
  background: #18181b;
  color: #f4f4f5;
  min-height: 140px;
}
.actor-active {
  border-color: #f59e0b;
  box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.35);
}
.avatar {
  font-size: 2rem;
  margin-bottom: 6px;
}
.actor-card small {
  display: block;
  margin-top: 6px;
  color: #d4d4d8;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Hackathon Actor Lab - Agentic Actors") as demo:
        state = gr.State(initial_session())
        gr.Markdown(
            "## Agentic Puppet Theater\n"
            "Actors have goals, secrets, tools, memory, and structured decisions. "
            "Try **Request audit**, then **Run one beat**."
        )
        stage = gr.HTML(render_stage(initial_session()))
        chat = gr.Chatbot(label="Stage transcript", height=340)
        with gr.Row():
            run = gr.Button("Run one beat", variant="primary")
            audit = gr.Button("Request audit")
            prop = gr.Textbox(label="Throw prop", value="rubber duck", scale=2)
            prop_btn = gr.Button("Throw")
        with gr.Row():
            summon_name = gr.Textbox(label="Summon actor", value="Oracle Spoon")
            summon = gr.Button("Summon")
            reset = gr.Button("Reset")
        with gr.Row():
            director = gr.Code(label="Director / policy log", language=None, lines=16)
            tools = gr.Code(label="Tool log", language=None, lines=16)

        outputs = [state, stage, chat, director, tools]
        demo.queue(default_concurrency_limit=20)
        run.click(run_one_beat, inputs=[state], outputs=outputs)
        audit.click(queue_audit, inputs=[state], outputs=outputs)
        prop_btn.click(queue_prop, inputs=[state, prop], outputs=outputs)
        summon.click(summon_actor, inputs=[state, summon_name], outputs=outputs)
        reset.click(reset_session, outputs=outputs)

    return demo


def main() -> None:
    build_app().launch(server_name="127.0.0.1", server_port=7874, show_error=True, css=CSS)


if __name__ == "__main__":
    main()
