"""Concept 15: agentic actors with basic text-to-speech.

Run:
    uv run python src/agentic_actor_gradio_tts.py

This extends the agentic actor Gradio demo with real text-to-speech via
`edge-tts` when available. If real TTS fails, it falls back to generated WAV
audio so the demo remains reliable.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import struct
import sys
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import gradio as gr

try:
    import edge_tts
except ImportError:  # pragma: no cover
    edge_tts = None

BASE_PATH = Path(__file__).with_name("agentic_actor_gradio.py")
SPEC = importlib.util.spec_from_file_location("agentic_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load src/agentic_actor_gradio.py")
agentic_base = importlib.util.module_from_spec(SPEC)
sys.modules["agentic_base"] = agentic_base
SPEC.loader.exec_module(agentic_base)
agentic_base.ActorDecision.model_rebuild()


AUDIO_DIR = Path(tempfile.mkdtemp(prefix="fastrack_tts_"))
EDGE_VOICE_BY_ACTOR = {
    "Sock Dragon": "en-US-GuyNeural",
    "Sir Croaksalot": "en-US-JennyNeural",
    "Cactus Accountant": "en-US-SteffanNeural",
    "TOOL": "en-US-AriaNeural",
}
VOICE_HZ = {
    "Sock Dragon": 95,
    "Sir Croaksalot": 260,
    "Cactus Accountant": 150,
    "TOOL": 420,
}


async def _edge_tts_save(text: str, voice: str, path: Path) -> None:
    assert edge_tts is not None
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(path))


def synth_edge_tts(who: str, text: str) -> str:
    """Generate actual spoken MP3 with edge-tts."""
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed")
    cleaned = (text or "").strip() or "..."
    voice = EDGE_VOICE_BY_ACTOR.get(who, "en-US-AriaNeural")
    path = AUDIO_DIR / f"edge_{who.replace(' ', '_')}_{uuid.uuid4().hex}.mp3"
    asyncio.run(_edge_tts_save(cleaned[:3000], voice, path))
    return str(path)


def synth_placeholder_audio(who: str, text: str) -> str:
    """Generate a tiny offline WAV fallback whose duration roughly follows text length."""
    sample_rate = 22_050
    base_hz = VOICE_HZ.get(who, 210)
    duration = min(4.5, max(0.5, len(text) / 38.0))
    samples_count = int(sample_rate * duration)
    samples = bytearray()

    for i in range(samples_count):
        t = i / sample_rate
        envelope = math.sin(math.pi * i / max(samples_count - 1, 1))
        wobble = 1.0 + 0.04 * math.sin(2 * math.pi * 5 * t)
        carrier = math.sin(2 * math.pi * base_hz * wobble * t)
        syllable = 0.18 * math.sin(2 * math.pi * 8 * t)
        value = (carrier * 0.65 + syllable * 0.15) * envelope
        samples.extend(struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767)))

    path = AUDIO_DIR / f"{who.replace(' ', '_')}_{uuid.uuid4().hex}.wav"
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples)
    return str(path)


def synth_line_audio(who: str, text: str) -> str:
    """Try real TTS first, then fall back to deterministic generated audio."""
    try:
        return synth_edge_tts(who, text)
    except Exception:
        return synth_placeholder_audio(who, text)


def wait_for_audio_playback(text: str) -> None:
    """Give the browser time to play the current audio before replacing it."""
    seconds = min(8.0, max(1.0, len(text) / 13.0 + 0.45))
    time.sleep(seconds)


def ui_payload(
    session: dict[str, Any],
    active: str | None = None,
    audio_path: str | None = None,
) -> tuple[Any, ...]:
    return (
        session,
        agentic_base.render_stage(session, active),
        agentic_base.transcript_chat(session),
        agentic_base.logs_text(session),
        agentic_base.tool_text(session),
        gr.update(value=audio_path, autoplay=True) if audio_path else gr.skip(),
    )


def run_one_beat_with_tts(session: dict[str, Any]):
    session = agentic_base.copy_session(session)
    agentic_base.apply_audience_queue(session)
    session["beat"] = int(session.get("beat", 0)) + 1

    speaker = agentic_base.current_speaker(session)
    listeners = agentic_base.listener_actors(session, speaker)
    agentic_base.append_director(
        session,
        f"[Director] Beat {session['beat']}: floor={speaker['name']}; "
        f"listeners={', '.join(a['name'] for a in listeners)}.",
    )
    yield ui_payload(session, speaker["name"])

    line = agentic_base.speaker_line(session, speaker)
    partial = ""
    for word in line.split():
        partial = f"{partial} {word}".strip()
        agentic_base.append_transcript(session, speaker["name"], partial)
        if len(session["transcript"]) >= 2:
            previous = session["transcript"][-2]
            if previous["who"] == speaker["name"] and partial.startswith(previous["text"]):
                session["transcript"].pop(-2)
        yield ui_payload(session, speaker["name"])
        time.sleep(0.08)

    yield ui_payload(session, speaker["name"], synth_line_audio(speaker["name"], line))
    wait_for_audio_playback(line)

    decisions = [
        agentic_base.decide_actor_action(session, actor, speaker, partial)
        for actor in listeners
    ]
    for decision in decisions:
        agentic_base.append_director(
            session,
            f"[{decision.actor} policy] {decision.action} "
            f"score={decision.score} intent={decision.intent}; reason={decision.reason}",
        )
    yield ui_payload(session, speaker["name"])

    tool_decisions = [d for d in decisions if d.action == "USE_TOOL" and d.score >= 70]
    for decision in tool_decisions[:1]:
        actor = next(a for a in session["actors"] if a["name"] == decision.actor)
        assert decision.tool_request is not None
        result = agentic_base.run_tool(session, actor, decision.tool_request)
        agentic_base.append_transcript(session, "TOOL", result)
        yield ui_payload(session, decision.actor, synth_line_audio("TOOL", result))
        wait_for_audio_playback(result)

        decisions = [
            agentic_base.decide_actor_action(session, actor, speaker, result)
            for actor in agentic_base.listener_actors(session, speaker)
        ]
        for followup in decisions:
            agentic_base.append_director(
                session,
                f"[{followup.actor} follow-up] {followup.action} "
                f"score={followup.score}; reason={followup.reason}",
            )
        yield ui_payload(session, decision.actor)

    interrupt = agentic_base.pick_best_interrupt(decisions)
    if interrupt is not None:
        agentic_base.append_director(
            session, f"[Director] Interrupt approved for {interrupt.actor}."
        )
        actor = next(a for a in session["actors"] if a["name"] == interrupt.actor)
        actor["cooldown"] = 1
        agentic_base.append_transcript(session, interrupt.actor, interrupt.dialogue)
        session["current_index"] = session["actors"].index(actor)
        yield ui_payload(
            session,
            interrupt.actor,
            synth_line_audio(interrupt.actor, interrupt.dialogue),
        )
        wait_for_audio_playback(interrupt.dialogue)
    else:
        agentic_base.append_director(
            session, "[Director] No interrupt approved; normal turn advance."
        )

    for actor in session["actors"]:
        actor["cooldown"] = max(0, int(actor.get("cooldown", 0)) - 1)
    session["current_index"] = (int(session["current_index"]) + 1) % len(session["actors"])
    yield ui_payload(session, None)


def passthrough_with_no_audio(result: tuple[Any, ...]) -> tuple[Any, ...]:
    return (*result, gr.skip())


def queue_audit(session: dict[str, Any]) -> tuple[Any, ...]:
    return passthrough_with_no_audio(agentic_base.queue_audit(session))


def queue_prop(session: dict[str, Any], prop: str) -> tuple[Any, ...]:
    return passthrough_with_no_audio(agentic_base.queue_prop(session, prop))


def summon_actor(session: dict[str, Any], name: str) -> tuple[Any, ...]:
    return passthrough_with_no_audio(agentic_base.summon_actor(session, name))


def reset_session() -> tuple[Any, ...]:
    return passthrough_with_no_audio(agentic_base.reset_session())


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Hackathon TTS Actor Lab - Agentic Actors With TTS") as demo:
        initial = agentic_base.initial_session()
        state = gr.State(initial)
        gr.Markdown(
            "## Agentic Puppet Theater With Basic TTS\n"
            "This is exercise 14 plus offline generated WAV audio. "
            "Try **Request audit**, then **Run one beat**."
        )
        stage = gr.HTML(agentic_base.render_stage(initial))
        chat = gr.Chatbot(label="Stage transcript", height=340)
        audio = gr.Audio(
            label="Current line audio",
            type="filepath",
            autoplay=True,
            interactive=False,
        )
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

        outputs = [state, stage, chat, director, tools, audio]
        demo.queue(default_concurrency_limit=20)
        run.click(run_one_beat_with_tts, inputs=[state], outputs=outputs)
        audit.click(queue_audit, inputs=[state], outputs=outputs)
        prop_btn.click(queue_prop, inputs=[state, prop], outputs=outputs)
        summon.click(summon_actor, inputs=[state, summon_name], outputs=outputs)
        reset.click(reset_session, outputs=outputs)

    return demo


def main() -> None:
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7875,
        show_error=True,
        css=agentic_base.CSS,
    )


if __name__ == "__main__":
    main()
