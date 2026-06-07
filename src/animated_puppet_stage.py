"""Concept 16: animated puppet-stage UI for agentic actors.

Run:
    uv run python src/animated_puppet_stage.py

This keeps the agentic actor + TTS engine from exercise 15, but presents it as
a stage-first puppet show instead of a dashboard.
"""

from __future__ import annotations

import html
import importlib.util
import sys
from pathlib import Path
from typing import Any

import gradio as gr

BASE_PATH = Path(__file__).with_name("agentic_actor_gradio_tts.py")
SPEC = importlib.util.spec_from_file_location("agentic_tts", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load src/agentic_actor_gradio_tts.py")
agentic_tts = importlib.util.module_from_spec(SPEC)
sys.modules["agentic_tts"] = agentic_tts
SPEC.loader.exec_module(agentic_tts)


def _latest_line_for_actor(session: dict[str, Any], actor_name: str) -> str:
    for item in reversed(session.get("transcript", [])):
        if item.get("who") == actor_name:
            return str(item.get("text", ""))
    return ""


def _last_actor_status(session: dict[str, Any], actor_name: str, active: str | None) -> str:
    if active == actor_name:
        return "speaking"
    logs = " ".join(session.get("director_log", [])[-8:])
    if (
        f"[{actor_name} follow-up] INTERRUPT" in logs
        or f"Interrupt approved for {actor_name}" in logs
    ):
        return "interrupting"
    if "[Tool:" in logs and actor_name in logs:
        return "using tool"
    if f"[{actor_name} policy]" in logs:
        return "listening"
    return "waiting"


def _stage_event_badge(session: dict[str, Any]) -> str:
    rules = session.get("reality_rules", [])
    if not rules:
        return "Stage reality: normal"
    return f"Stage reality: {html.escape(str(rules[-1]))}"


def _prop_emoji(prop: str) -> str:
    text = prop.lower()
    if "duck" in text:
        return "🦆"
    if "potato" in text:
        return "🥔"
    if "crown" in text:
        return "👑"
    if "spoon" in text:
        return "🥄"
    if "rock" in text:
        return "🪨"
    return "🎁"


def render_theater_stage(session: dict[str, Any], active: str | None = None) -> str:
    cards = []
    for actor in session["actors"]:
        name = str(actor["name"])
        escaped_name = html.escape(name)
        avatar = html.escape(str(actor["avatar"]))
        goal = html.escape(str(actor["goal"]))
        tools = html.escape(", ".join(actor.get("tools", [])) or "none")
        latest = html.escape(_latest_line_for_actor(session, name) or "...")
        status = _last_actor_status(session, name, active)
        status_class = status.replace(" ", "-")
        active_class = " is-active" if active == name else ""
        cards.append(
            f"""
<div class="puppet-slot {status_class}{active_class}">
  <div class="speech-bubble">{latest}</div>
  <div class="strings"><i></i><i></i><i></i></div>
  <div class="puppet-body">
    <div class="puppet-avatar">{avatar}</div>
    <div class="puppet-name">{escaped_name}</div>
    <div class="puppet-status">{html.escape(status)}</div>
  </div>
  <div class="puppet-meta">
    <span>Goal: {goal}</span>
    <span>Tools: {tools}</span>
  </div>
</div>
"""
        )

    beat = int(session.get("beat", 0))
    cast_size = len(session.get("actors", []))
    thrown_prop = str(session.get("last_thrown_prop") or "")
    flying_prop = (
        f'<div class="flying-prop" aria-label="Thrown prop">{_prop_emoji(thrown_prop)}</div>'
        if thrown_prop
        else ""
    )
    return f"""
<section class="theater-shell">
  <div class="marquee">
    <span>The Runaway Puppet Show</span>
    <small>Beat {beat} · Cast {cast_size}</small>
  </div>
  <div class="curtain curtain-left"></div>
  <div class="curtain curtain-right"></div>
  <div class="stage-lights"><i></i><i></i><i></i></div>
  {flying_prop}
  <div class="stage-event">{_stage_event_badge(session)}</div>
  <div class="puppet-stage">
    {"".join(cards)}
  </div>
  <div class="stage-floor"></div>
</section>
"""


agentic_tts.agentic_base.render_stage = render_theater_stage


def queue_prop_with_animation(session: dict[str, Any], prop: str) -> tuple[Any, ...]:
    session = agentic_tts.agentic_base.copy_session(session)
    payload = prop.strip() or "rubber duck"
    session["last_thrown_prop"] = payload
    session["audience_queue"].append({"kind": "prop", "payload": payload})
    agentic_tts.agentic_base.append_director(
        session, f"[Audience] Throws {payload}; it arcs across the stage."
    )
    return agentic_tts.ui_payload(session)


def summon_actor_immediately(session: dict[str, Any], name: str) -> tuple[Any, ...]:
    session = agentic_tts.agentic_base.copy_session(session)
    cast_limit = 5
    if len(session["actors"]) >= cast_limit:
        agentic_tts.agentic_base.append_director(
            session, f"[Director] Summon rejected: stage limit is {cast_limit} puppets."
        )
        return agentic_tts.ui_payload(session)

    actor_name = name.strip() or "Oracle Spoon"
    new_actor = {
        "name": actor_name,
        "avatar": _prop_emoji(actor_name),
        "goal": "Make prophecies that complicate the current scene",
        "secret": "Can hear stage directions before they happen",
        "style": "paranoid, prophetic, theatrical",
        "tools": ["prop_oracle", "memory_lookup"],
        "memory": [],
        "cooldown": 0,
    }
    session["actors"].append(new_actor)
    session["reality_rules"].append(f"{actor_name} has joined the performance.")
    agentic_tts.agentic_base.append_director(
        session, f"[Audience] Summons new actor immediately: {actor_name}."
    )
    return agentic_tts.ui_payload(session, actor_name)


def reset_session() -> tuple[Any, ...]:
    return agentic_tts.ui_payload(agentic_tts.agentic_base.initial_session())


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Hackathon Animated Puppet Stage - Animated Puppet Stage") as demo:
        initial = agentic_tts.agentic_base.initial_session()
        state = gr.State(initial)
        gr.Markdown(
            "## The Runaway Puppet Show\n"
            "A stage-first version of the agentic actor demo. "
            "Try **Request audit**, then **Run one beat** to trigger tool use and interruption."
        )
        stage = gr.HTML(render_theater_stage(initial))
        with gr.Row():
            run = gr.Button("Run one beat", variant="primary")
            audit = gr.Button("Request audit")
            prop = gr.Textbox(label="Throw prop", value="rubber duck", scale=2)
            prop_btn = gr.Button("Throw")
        with gr.Row():
            summon_name = gr.Textbox(label="Summon actor", value="Oracle Spoon")
            summon = gr.Button("Summon")
            reset = gr.Button("Reset")
        audio = gr.Audio(
            label="Puppet voice",
            type="filepath",
            autoplay=True,
            interactive=False,
        )
        with gr.Accordion("Transcript", open=True):
            chat = gr.Chatbot(label="Stage transcript", height=260)
        with gr.Accordion("Behind the curtain", open=False):
            with gr.Row():
                director = gr.Code(label="Director / policy log", language=None, lines=16)
                tools = gr.Code(label="Tool log", language=None, lines=16)

        outputs = [state, stage, chat, director, tools, audio]
        demo.queue(default_concurrency_limit=20)
        run.click(agentic_tts.run_one_beat_with_tts, inputs=[state], outputs=outputs)
        audit.click(agentic_tts.queue_audit, inputs=[state], outputs=outputs)
        prop_btn.click(queue_prop_with_animation, inputs=[state, prop], outputs=outputs)
        summon.click(summon_actor_immediately, inputs=[state, summon_name], outputs=outputs)
        reset.click(reset_session, outputs=outputs)

    return demo


CSS = """
.gradio-container {
  background:
    radial-gradient(1200px 700px at 50% -10%, #2b1018 0%, #11070b 52%, #070304 100%)
    !important;
}
.theater-shell {
  position: relative;
  overflow: hidden;
  min-height: 520px;
  border-radius: 12px;
  border: 2px solid rgba(245, 190, 120, 0.28);
  background:
    radial-gradient(circle at 50% 14%, rgba(255, 216, 135, 0.18), transparent 30%),
    linear-gradient(180deg, #24100d 0%, #10080a 56%, #1b0c07 100%);
  box-shadow: 0 22px 70px rgba(0, 0, 0, 0.55), inset 0 0 0 1px rgba(255, 255, 255, 0.04);
}
.marquee {
  position: relative;
  z-index: 4;
  text-align: center;
  padding: 16px 12px 12px;
  color: #ffe8bd;
  font-family: Georgia, serif;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  background: linear-gradient(180deg, rgba(52, 26, 10, 0.92), rgba(34, 16, 8, 0.78));
  border-bottom: 1px solid rgba(255, 220, 160, 0.2);
}
.marquee span { display: block; font-size: 1.2rem; }
.marquee small {
  display: block;
  margin-top: 4px;
  color: #dbb47e;
  letter-spacing: 0;
  text-transform: none;
}
.curtain {
  position: absolute;
  top: 0;
  bottom: 68px;
  z-index: 2;
  width: 9%;
  background:
    repeating-linear-gradient(90deg, rgba(255,255,255,0.08) 0 3px, transparent 3px 22px),
    linear-gradient(90deg, #4a0612, #8c1230 48%, #520717);
  box-shadow: inset 0 0 35px rgba(0,0,0,0.65);
}
.curtain-left { left: 0; }
.curtain-right { right: 0; transform: scaleX(-1); }
.stage-lights {
  position: absolute;
  inset: 70px 12% auto;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  pointer-events: none;
}
.stage-lights i {
  height: 260px;
  background:
    radial-gradient(
      ellipse at 50% 0%,
      rgba(255, 217, 145, 0.34),
      rgba(255, 217, 145, 0.06) 48%,
      transparent 72%
    );
  filter: blur(1px);
}
.flying-prop {
  position: absolute;
  z-index: 5;
  top: 170px;
  left: -40px;
  font-size: 2.6rem;
  filter: drop-shadow(0 10px 14px rgba(0,0,0,0.55));
  animation: throw-arc 1.1s cubic-bezier(0.2, 0.7, 0.25, 1) both;
  pointer-events: none;
}
.stage-event {
  position: relative;
  z-index: 3;
  width: fit-content;
  margin: 14px auto 4px;
  padding: 6px 12px;
  border-radius: 999px;
  color: #ffe3ad;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 212, 140, 0.22);
  font-size: 0.88rem;
}
.puppet-stage {
  position: relative;
  z-index: 3;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 16px;
  align-items: end;
  padding: 24px 11% 82px;
}
.puppet-slot {
  min-width: 0;
  text-align: center;
  transform-origin: 50% 100%;
  transition: transform 0.24s ease, filter 0.24s ease;
}
.puppet-slot.is-active {
  animation: puppet-bob 0.72s ease-in-out infinite;
  filter: drop-shadow(0 0 22px rgba(255, 198, 104, 0.55));
}
.puppet-slot.interrupting {
  animation: interrupt-pop 0.42s ease-out 2;
}
.puppet-slot.using-tool .puppet-body {
  box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.55), 0 0 28px rgba(96, 165, 250, 0.35);
}
.speech-bubble {
  min-height: 72px;
  max-height: 120px;
  overflow: hidden;
  padding: 10px 12px;
  border-radius: 10px;
  color: #fff6e7;
  background: rgba(20, 10, 8, 0.82);
  border: 1px solid rgba(255, 219, 165, 0.22);
  box-shadow: 0 10px 24px rgba(0,0,0,0.32);
  font-size: 0.92rem;
  line-height: 1.32;
}
.strings {
  display: flex;
  justify-content: center;
  gap: 28px;
  height: 42px;
  opacity: 0.55;
}
.strings i {
  width: 1px;
  background: linear-gradient(180deg, rgba(255, 236, 198, 0.7), rgba(255, 236, 198, 0));
}
.puppet-body {
  width: min(160px, 82%);
  margin: 0 auto;
  padding: 14px 12px;
  border-radius: 12px 12px 18px 18px;
  color: #fff2dc;
  background:
    radial-gradient(circle at 35% 12%, rgba(255,255,255,0.12), transparent 28%),
    linear-gradient(180deg, #402017, #1c0d0a);
  border: 1px solid rgba(255, 213, 155, 0.22);
  box-shadow: 0 16px 38px rgba(0,0,0,0.42);
}
.puppet-avatar { font-size: 2.4rem; line-height: 1; margin-bottom: 8px; }
.puppet-name { font-weight: 800; }
.puppet-status {
  display: inline-block;
  margin-top: 8px;
  padding: 3px 8px;
  border-radius: 999px;
  color: #1f1305;
  background: #f4c56f;
  font-size: 0.74rem;
  text-transform: uppercase;
}
.puppet-meta {
  margin-top: 10px;
  display: grid;
  gap: 4px;
  color: #d8c2a2;
  font-size: 0.78rem;
}
.stage-floor {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 78px;
  z-index: 2;
  background:
    repeating-linear-gradient(90deg, rgba(255,255,255,0.05) 0 2px, transparent 2px 42px),
    linear-gradient(180deg, #6f421d, #2b1609);
  border-top: 2px solid rgba(255, 211, 140, 0.18);
}
@keyframes puppet-bob {
  0%, 100% { transform: translateY(0) rotate(-0.5deg); }
  50% { transform: translateY(-8px) rotate(0.5deg); }
}
@keyframes interrupt-pop {
  0% { transform: scale(1); }
  35% { transform: scale(1.08) rotate(-2deg); }
  70% { transform: scale(0.99) rotate(1deg); }
  100% { transform: scale(1); }
}
@keyframes throw-arc {
  0% { transform: translate(-20px, 90px) rotate(-30deg); opacity: 0; }
  12% { opacity: 1; }
  55% { transform: translate(48vw, -45px) rotate(180deg); opacity: 1; }
  100% { transform: translate(94vw, 84px) rotate(390deg); opacity: 0; }
}
"""


def main() -> None:
    build_app().launch(server_name="127.0.0.1", server_port=7876, show_error=True, css=CSS)


if __name__ == "__main__":
    main()
