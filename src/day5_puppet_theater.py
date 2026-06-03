"""
AI Puppet Theater With Tiny Actors — Gradio 6 demo.

- Dark theatrical UI (Soft theme + velvet CSS), Backstage vs Main Stage.
- Curtain drop + intro chime, then unlocks the stage.
- Clause streaming + local **Ollama** dialogue; **edge-tts** neural speech per clause (MP3).
- Three voice presets, glowing speaker, dimmed listeners, interruption scoring + gasp clip.
- Director logs, Play / Pause / Reset (shared in-session control dict; use queue concurrency).

Run: uv run python src/day5_puppet_theater.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import html
import math
import os
import re
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr
from ollama import Client as OllamaClient

# ---------------------------------------------------------------------------
# Optional: Hugging Face InferenceClient — install `huggingface_hub` and wire
# your endpoint/model in ``llm_stream_turn`` / ``tts_synthesize``.
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import InferenceClient  # type: ignore
except ImportError:  # pragma: no cover - offline / minimal env
    InferenceClient = None  # type: ignore[misc, assignment]

try:
    import edge_tts  # type: ignore
except ImportError:  # pragma: no cover
    edge_tts = None  # type: ignore[misc, assignment]


# --- Global stream epoch (same pattern as Day 3: preempt running generators) ---
_stream_epoch = [0]
_epoch_lock = threading.Lock()


def _bump_stream_epoch() -> int:
    with _epoch_lock:
        _stream_epoch[0] += 1
        return _stream_epoch[0]


def _current_epoch() -> int:
    return _stream_epoch[0]


# --- Audio asset cache (WAV files in temp dir) ---
_AUDIO_DIR = Path(tempfile.mkdtemp(prefix="puppet_theater_audio_"))

VOICE_PRESETS: dict[str, dict[str, Any]] = {
    "Deep/Dramatic": {"base_hz": 95, "wobble": 0.02, "kind": "sine"},
    "High-pitched/Nervous": {"base_hz": 280, "wobble": 0.35, "kind": "fm"},
    "Robotic/Monotone": {"base_hz": 160, "wobble": 0.0, "kind": "square"},
}

# Microsoft Edge neural voices (edge-tts) — one per tone preset; override with EDGE_TTS_VOICE_* env vars.
EDGE_TTS_VOICE_BY_TONE: dict[str, str] = {
    "Deep/Dramatic": os.environ.get("EDGE_TTS_VOICE_DEEP", "en-US-GuyNeural"),
    "High-pitched/Nervous": os.environ.get("EDGE_TTS_VOICE_NERVOUS", "en-US-JennyNeural"),
    "Robotic/Monotone": os.environ.get("EDGE_TTS_VOICE_ROBOT", "en-US-SteffanNeural"),
}
DEFAULT_EDGE_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-AriaNeural")


async def _edge_tts_save_mp3(text: str, voice: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def synth_edge_tts(text: str, voice_label: str) -> str:
    """Neural TTS via edge-tts → MP3 path. Requires network. Runs async in a worker thread."""
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed")
    clean = (text or "").strip() or "…"
    if len(clean) > 4000:
        clean = clean[:3997] + "…"
    voice = EDGE_TTS_VOICE_BY_TONE.get(voice_label, DEFAULT_EDGE_VOICE)
    out = _AUDIO_DIR / f"edge_{uuid.uuid4().hex}.mp3"

    async def _go() -> None:
        await _edge_tts_save_mp3(clean, voice, out)

    def _run_loop() -> None:
        asyncio.run(_go())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_run_loop).result(timeout=120)
    return str(out)


def _write_wav_mono(path: Path, samples: list[float], sample_rate: int = 22050) -> str:
    """Write float samples in [-1, 1] to 16-bit mono WAV."""
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for x in samples:
            x = max(-1.0, min(1.0, x))
            frames.extend(struct.pack("<h", int(x * 32767 * 0.72)))
        wf.writeframes(frames)
    return str(path)


def synth_placeholder_tts(text: str, voice_label: str, duration_cap: float = 6.0) -> str:
    """
    Fast offline TTS stand-in: amplitude-modulated tone whose length scales with text.

    Replace with InferenceClient text-to-speech (or your HF Space) and return a filepath/URL.
    """
    preset = VOICE_PRESETS.get(voice_label, VOICE_PRESETS["Robotic/Monotone"])
    sr = 22050
    base = float(preset["base_hz"])
    kind = str(preset["kind"])
    wobble = float(preset["wobble"])
    dur = min(duration_cap, 0.35 + min(4.5, len(text) / 42.0))
    n = int(sr * dur)
    samples: list[float] = []
    for i in range(n):
        t = i / sr
        carrier = math.sin(2 * math.pi * base * t)
        if kind == "fm":
            mod = 1.0 + wobble * math.sin(2 * math.pi * 7.0 * t)
            carrier = math.sin(2 * math.pi * base * mod * t)
        elif kind == "square":
            carrier = 1.0 if math.sin(2 * math.pi * base * t) >= 0 else -1.0
        else:
            carrier = math.sin(2 * math.pi * base * (1.0 + wobble * math.sin(2 * math.pi * 3 * t)) * t)
        # Cheap "speech-ish" envelope + syllabic bumps from text hash
        h = (hash(text) % 1000) / 1000.0
        env = 0.55 + 0.45 * math.sin(math.pi * t / max(dur, 1e-6))
        bump = 0.15 * math.sin(2 * math.pi * (4.0 + h * 3.0) * t)
        samples.append(carrier * env + bump * 0.05)

    out = _AUDIO_DIR / f"tts_{uuid.uuid4().hex}.wav"
    return _write_wav_mono(out, samples, sr)


def synth_intro_chime() -> str:
    """Short theater intro (three descending tones)."""
    sr = 22050
    tones = [(880, 0.12), (660, 0.12), (523, 0.22)]
    samples: list[float] = []
    for freq, seg in tones:
        n = int(sr * seg)
        for i in range(n):
            t = i / sr
            env = math.sin(math.pi * (i + 1) / max(n, 1))
            samples.append(env * math.sin(2 * math.pi * freq * t) * 0.55)
    out = _AUDIO_DIR / "intro_chime.wav"
    return _write_wav_mono(out, samples, sr)


def synth_gasp() -> str:
    """Tiny interruption sting."""
    sr = 22050
    dur = 0.18
    n = int(sr * dur)
    samples = [math.sin(2 * math.pi * 420 * (i / sr)) * math.exp(-6 * i / max(n, 1)) * 0.7 for i in range(n)]
    out = _AUDIO_DIR / "gasp.wav"
    return _write_wav_mono(out, samples, sr)


INTRO_CHIME_PATH = synth_intro_chime()
GASP_PATH = synth_gasp()


def _wav_duration_seconds(path: str) -> float | None:
    try:
        with wave.open(path, "r") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return None


def _ffprobe_duration_seconds(path: str) -> float | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return float(proc.stdout.strip())
    except Exception:
        return None
    return None


def estimated_playback_seconds(audio_path: str, text_for_heuristic: str = "") -> float:
    """Best-effort audio length so we don't start the next clip before this one finishes."""
    p = audio_path.lower()
    if p.endswith(".wav"):
        d = _wav_duration_seconds(audio_path)
        if d is not None:
            return max(0.25, d + 0.35)
    d = _ffprobe_duration_seconds(audio_path)
    if d is not None:
        return max(0.3, d + 0.45)
    # MP3 / unknown: conservative chars/sec + pad (avoids cutting off neural TTS early)
    base = max(1.0, min(95.0, len(text_for_heuristic) / 11.0 + 0.75))
    return base


def _sleep_until_playback_end(
    audio_path: str,
    text_for_heuristic: str,
    session: dict[str, Any],
    run_epoch: int,
) -> bool:
    """
    Block until the clip should have finished (avoids replacing ``line_audio`` while still playing).
    Returns False if epoch changed or stop requested during wait.
    """
    total = estimated_playback_seconds(audio_path, text_for_heuristic)
    deadline = time.monotonic() + total
    while time.monotonic() < deadline:
        if _current_epoch() != run_epoch or session["_ctrl"]["stop"]:
            return False
        while session["_ctrl"]["pause"] and not session["_ctrl"]["stop"]:
            time.sleep(0.1)
        if session["_ctrl"]["stop"]:
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.12, remaining))
    return True


def _split_clauses(text: str) -> list[str]:
    """Split on sentence end or em-dash / semicolon for streaming granularity."""
    parts = re.split(r"(?<=[.!?])\s+|;\s+| — ", text.strip())
    return [p.strip() for p in parts if p.strip()]


DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
MIN_CLAUSES_BEFORE_INTERRUPT = 4
INTERRUPT_SCORE_THRESHOLD = 82


def _ollama_reachable(host: str | None) -> bool:
    try:
        OllamaClient(host=host).list()
        return True
    except Exception:
        return False


def _format_transcript(dialogue: list[dict[str, Any]]) -> str:
    lines = []
    for turn in dialogue[-24:]:
        name = turn.get("name", "Actor")
        text = (turn.get("text") or "").strip()
        if text:
            lines.append(f"{name}: {text}")
    return "\n".join(lines) if lines else "(The scene has just opened — no lines yet.)"


def _pull_complete_sentences(buf: str) -> tuple[list[str], str]:
    """Split leading complete sentences from buffer; return (sentences, remainder)."""
    out: list[str] = []
    while True:
        m = re.search(r"(?<=[.!?])\s+", buf)
        if not m:
            break
        sent = buf[: m.end()].strip()
        buf = buf[m.end() :].lstrip()
        if sent:
            out.append(sent)
    if len(buf) > 280:
        cut = buf.rfind(", ", 80, len(buf))
        if cut == -1:
            cut = buf.rfind(" ", 80, len(buf))
        if cut > 40:
            out.append(buf[:cut].strip())
            buf = buf[cut + 1 :].lstrip()
    return out, buf


PLAYWRIGHT_SYSTEM = """You are the voice of ONE character in a short puppet-theater scene.
Rules:
- Write ONLY spoken dialogue for that character (first person). No stage directions, no asterisks, no labels like "Actor 1:".
- Stay consistent with the character's goal and speaking tone; their secret is private — hint at tension but do not dump the secret verbatim unless dramatically necessary.
- 3–7 sentences total, natural conversational pace. Vary length; react to what others said in the transcript."""


def _playwright_user_prompt(
    environment: str,
    topic: str,
    chars: list[dict[str, str]],
    actor_index: int,
    transcript: str,
) -> str:
    roster = []
    for i, ch in enumerate(chars):
        roster.append(
            f"- {ch.get('name', f'Actor {i+1}')} (tone: {ch.get('tone','')}) — goal: {ch.get('goal','')}; "
            f"secret (for you only, do not quote): {ch.get('secret','')}"
        )
    name = chars[actor_index].get("name", f"Actor {actor_index + 1}")
    return (
        f"Setting: **{environment}**. Topic: **{topic}**.\n\n"
        f"Characters:\n" + "\n".join(roster) + "\n\n"
        f"Transcript so far:\n{transcript}\n\n"
        f"Write the next lines for **{name}** only. They hold the floor; respond to the others and advance the scene."
    )


def _stream_ollama_turn(
    session: dict[str, Any],
    actor_index: int,
    model: str,
    host: str | None,
    environment: str,
    topic: str,
    chars: list[dict[str, str]],
) -> Iterator[str]:
    dialogue = session.get("dialogue") or []
    transcript = _format_transcript(dialogue)
    user = _playwright_user_prompt(environment, topic, chars, actor_index, transcript)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": PLAYWRIGHT_SYSTEM},
        {"role": "user", "content": user},
    ]
    buf = ""
    yielded_one = False
    try:
        stream = OllamaClient(host=host).chat(model=model, messages=messages, stream=True)
        for part in stream:
            piece = ""
            if getattr(part, "message", None) is not None and part.message.content:
                piece = part.message.content
            buf += piece
            flushed, buf = _pull_complete_sentences(buf)
            for sent in flushed:
                yielded_one = True
                yield sent.strip()
        buf = buf.strip()
        if buf:
            for chunk in _split_clauses(buf):
                if chunk:
                    yielded_one = True
                    yield chunk
        if not yielded_one:
            yield "Listen — we need one honest sentence before this room detonates."
    except Exception as exc:  # pragma: no cover - network / daemon
        session.setdefault("_llm_errors", []).append(str(exc))
        yield from _stream_mock_turn(session, actor_index, environment, topic, chars, reason=f"Ollama error ({exc!r}); using offline lines.")


def _stream_mock_turn(
    session: dict[str, Any],
    actor_index: int,
    environment: str,
    topic: str,
    chars: list[dict[str, str]],
    *,
    reason: str = "",
) -> Iterator[str]:
    """Short canned multi-clause lines so three actors rotate when Ollama is off."""
    _ = reason
    ch = chars[actor_index]
    goal = (ch.get("goal") or "")[:100]
    tcount = len(session.get("dialogue") or [])
    lines = [
        f"Alright — in this {environment.lower()}, about {topic}, I'm holding onto one fact: {goal}.",
        "Give me a straight answer: what changed since we last talked?",
        "If we're honest, the room already knows something's off; I just need the next step.",
    ]
    if actor_index == 1:
        lines = [
            f"I hear you, but panic won't fix {topic.lower()}. We breathe, we sequence the risks.",
            "I'll say it once: nobody leaves until we agree who owns the awkward truth.",
            "Can we table the blame for sixty seconds and pick one move?",
        ]
    elif actor_index == 2:
        lines = [
            f"Logging this as {environment} incident thread; objective is to reduce harm, not win a speech contest.",
            "Data point: two conflicting stories. I need one timeline we can sign.",
            "If we can't align, I recommend a pause and written recap — otherwise variance explodes.",
        ]
    twist = (tcount + actor_index) % 3
    if twist == 0:
        lines.append("And before anyone grandstands — check the exits. Metaphorically.")
    elif twist == 1:
        lines.append("Small promise: I'll repeat back what I think I heard, then you correct me.")
    else:
        lines.append("End beat: we either commit to a plan or we reschedule with stakeholders.")
    for sent in lines:
        yield sent


def llm_stream_turn(
    session: dict[str, Any],
    hf_client: Any | None,
    actor_index: int,
    actor_name: str,
    goal: str,
    secret: str,
    tone: str,
    environment: str,
    topic: str,
    other_bubbles: list[str],
) -> Iterator[str]:
    """
    Stream one character's turn as complete sentences/clauses (for TTS per clause).

    Uses **local Ollama** when ``session['use_ollama']`` is true and the daemon is reachable;
    otherwise falls back to ``_stream_mock_turn``. ``session['dialogue']`` holds prior lines.
    """
    _ = hf_client, goal, secret, tone, other_bubbles, actor_name
    chars = session["characters"]
    use_ollama = bool(session.get("use_ollama", True))
    model = (session.get("ollama_model") or DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
    host = os.environ.get("OLLAMA_HOST")

    if use_ollama and _ollama_reachable(host):
        yield from _stream_ollama_turn(session, actor_index, model, host, environment, topic, chars)
    else:
        if use_ollama:
            _log(
                session,
                "[LLM] Ollama unreachable — start `ollama serve` and `ollama pull "
                f"{model}`, or disable **Use Ollama** for canned lines.",
            )
        yield from _stream_mock_turn(session, actor_index, environment, topic, chars)


def tts_synthesize(client: Any | None, text: str, voice_label: str) -> str:
    """
    Text-to-speech for each spoken clause: **edge-tts** (neural MP3) when available,
    else the tone-based placeholder WAV. Optional HF ``client`` can replace this later.

    Example (HF)::

        out_wav = client.text_to_speech(text, model=\"espnet/kan-bayashi_ljspeech_vits\")
        Path(\"/tmp/line.wav\").write_bytes(out_wav)
        return str(Path(\"/tmp/line.wav\"))

    """
    _ = client
    try:
        return synth_edge_tts(text, voice_label)
    except Exception:
        return synth_placeholder_tts(text, voice_label)


# --- UI rendering ---

THEATER_CSS = """
.gradio-container { background: radial-gradient(1200px 800px at 50% 0%, #2a1020 0%, #0c0608 55%, #050308 100%) !important; }
.gr-block.gr-form { background: rgba(18, 6, 12, 0.92) !important; border: 1px solid #4a1a2a !important; border-radius: 12px !important; }
footer {opacity: 0.35;}
#curtain-root { min-height: 220px; border-radius: 10px; overflow: hidden; position: relative; }
.curtain-panel {
  position: absolute; inset: 0; display: flex; pointer-events: none;
}
.curtain-half {
  flex: 1;
  background: linear-gradient(90deg, #4c0000 0%, #7a0a1a 40%, #5c0012 100%);
  box-shadow: inset 0 0 60px rgba(0,0,0,0.65);
  transform: translateY(0);
  transition: transform 2.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.curtain-half.left { border-right: 2px solid rgba(255,215,0,0.25); }
.curtain-half.right { border-left: 2px solid rgba(255,215,0,0.25); }
.curtain-raised .curtain-half.left { transform: translateY(-100%); }
.curtain-raised .curtain-half.right { transform: translateY(-100%); }
#stage-wrap {
  padding: 16px; border-radius: 12px; border: 1px solid #3d1530; background: rgba(10,4,8,0.92);
  display: flex; flex-wrap: wrap; justify-content: center; align-items: flex-start; gap: 18px;
}
.actor-card {
  display: flex; flex-direction: column; align-items: center; flex: 1 1 200px;
  max-width: 32%; min-width: 170px; text-align: center;
}
.bubble {
  min-height: 72px; padding: 10px 12px; border-radius: 14px;
  background: #1a0d14; border: 1px solid #5c2a44; color: #f4e6ef;
  font-size: 0.95rem; line-height: 1.35; margin-bottom: 10px; width: 100%;
  box-shadow: 0 6px 24px rgba(0,0,0,0.45);
}
.bubble.dim { opacity: 0.5; filter: grayscale(0.2); }
.avatar {
  font-size: 2.4rem; border-radius: 50%; padding: 10px;
  border: 3px solid #3a1a2a; background: rgba(40,10,30,0.6);
  transition: box-shadow 0.25s ease, border-color 0.25s ease, transform 0.25s ease;
}
.avatar.speaking {
  border-color: #ffd66b;
  box-shadow: 0 0 22px rgba(255, 200, 120, 0.85), 0 0 42px rgba(255, 120, 80, 0.35);
  transform: scale(1.05);
}
.actor-label { font-size: 0.75rem; color: #cfa7bc; margin-top: 6px; }
"""

THEATER_THEME = gr.themes.Soft(
    primary_hue="rose",
    secondary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Source Sans 3"), "ui-sans-serif", "system-ui"],
).set(
    body_background_fill_dark="#0a0508",
    block_background_fill_dark="#140a10",
    block_border_width="1px",
    block_label_text_size="md",
)


def render_curtain_html(raised: bool) -> str:
    cls = "curtain-raised" if raised else ""
    return f"""
<div id="curtain-root" class="{cls}">
  <div class="curtain-panel">
    <div class="curtain-half left"></div>
    <div class="curtain-half right"></div>
  </div>
  <div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#f6e0c5; font-family: Georgia, serif; letter-spacing:0.12em; text-shadow:0 2px 12px #000;">
    <span style="opacity:0.85;">AI PUPPET THEATER</span>
  </div>
</div>
"""


def render_stage_html(
    names: list[str],
    avatars: list[str],
    bubbles: list[str],
    speaking: int,
) -> str:
    cards = []
    for i in range(3):
        dim = "dim" if speaking >= 0 and i != speaking else ""
        listen = "…" if speaking >= 0 and i != speaking and not bubbles[i].strip() else ""
        raw = bubbles[i] if bubbles[i].strip() else listen
        content = html.escape(raw)
        sp = "speaking" if i == speaking else ""
        cards.append(
            f"""
<div class="actor-card">
  <div class="bubble {dim}">{content}</div>
  <div class="avatar {sp}">{avatars[i]}</div>
  <div class="actor-label">{names[i]}</div>
</div>
            """
        )
    inner = "".join(cards)
    return f'<div id="stage-wrap">{inner}</div>'


def _log(session: dict[str, Any], line: str) -> None:
    logs: list[str] = session.setdefault("logs", [])
    stamp = time.strftime("%H:%M:%S")
    logs.append(f"[{stamp}] {line}")
    session["logs"] = logs[-400:]


def _ctrl_pause(session: dict[str, Any], paused: bool) -> dict[str, Any]:
    session["_ctrl"]["pause"] = bool(paused)
    return session


def _ctrl_stop(session: dict[str, Any]) -> dict[str, Any]:
    session["_ctrl"]["stop"] = True
    session["_ctrl"]["pause"] = False
    _bump_stream_epoch()
    return session


def _interrupt_scores(
    speaker_idx: int,
    line: str,
    chars: list[dict[str, str]],
    clause_idx: int,
    beat: int,
) -> tuple[int, int]:
    """Listener desire to interrupt; softened so first clauses don't always ping-pong."""
    others = [j for j in range(3) if j != speaker_idx]
    scores = []
    for j in others:
        c = chars[j]
        raw = f"{j}|{clause_idx}|{beat}|{line[:72]}".encode()
        h = int(hashlib.sha256(raw).hexdigest()[:8], 16)
        base = h % 100
        bonus = 6 if c.get("secret", "").lower()[:4] in line.lower() else 0
        g = [w for w in c.get("goal", "").lower().split() if len(w) > 4]
        goal_hits = sum(1 for w in g if w in line.lower())
        bonus += min(10, goal_hits * 3)
        scores.append(min(100, base + bonus))
    return scores[0], scores[1]


def initial_session() -> dict[str, Any]:
    return {
        "curtain_raised": False,
        "environment": "Garden",
        "topic": "A missing invitation",
        "characters": [
            {
                "name": "Actor 1",
                "avatar": "🎭",
                "goal": "Host the perfect garden party",
                "secret": "Forgot to send half the invites",
                "tone": "Deep/Dramatic",
            },
            {
                "name": "Actor 2",
                "avatar": "🦊",
                "goal": "Keep everyone calm",
                "secret": "Knows the caterer canceled",
                "tone": "High-pitched/Nervous",
            },
            {
                "name": "Actor 3",
                "avatar": "🤖",
                "goal": "Log outcomes objectively",
                "secret": "Was told to spin the report",
                "tone": "Robotic/Monotone",
            },
        ],
        "bubbles": ["", "", ""],
        "dialogue": [],
        "logs": [],
        "hf_llm": None,
        "hf_tts": None,
        "_ctrl": {"pause": False, "stop": False},
    }


def pack_session_from_ui(
    env: str,
    topic: str,
    a1_av: str,
    a1_goal: str,
    a1_sec: str,
    a1_tone: str,
    a2_av: str,
    a2_goal: str,
    a2_sec: str,
    a2_tone: str,
    a3_av: str,
    a3_goal: str,
    a3_sec: str,
    a3_tone: str,
    session: dict[str, Any],
) -> dict[str, Any]:
    ctrl = session.setdefault("_ctrl", {"pause": False, "stop": False})
    s = copy.deepcopy(session)
    s["_ctrl"] = ctrl
    s["environment"] = env or "Garden"
    s["topic"] = (topic or "Untitled scene").strip()
    s["characters"] = [
        {
            "name": "Actor 1",
            "avatar": a1_av,
            "goal": a1_goal,
            "secret": a1_sec,
            "tone": a1_tone,
        },
        {
            "name": "Actor 2",
            "avatar": a2_av,
            "goal": a2_goal,
            "secret": a2_sec,
            "tone": a2_tone,
        },
        {
            "name": "Actor 3",
            "avatar": a3_av,
            "goal": a3_goal,
            "secret": a3_sec,
            "tone": a3_tone,
        },
    ]
    s.setdefault("logs", [])
    s["dialogue"] = []
    return s


def raise_curtain_flow(
    session: dict[str, Any],
) -> Iterator[tuple[Any, ...]]:
    """
    Curtain animation (HTML/CSS), intro chime, then reveal Main Stage + switch tab.

    Uses ``gr.update`` so outputs bind to live components. ``curtain_raised`` becomes True
    only after the animation so **Play** stays gated until the stage is ready.
    """
    epoch = _current_epoch()
    s = session
    s.setdefault("_ctrl", {"pause": False, "stop": False})
    s["_ctrl"]["stop"] = False
    s["_ctrl"]["pause"] = False
    s["curtain_raised"] = False
    _log(s, "[Director] Raise Curtain — animating curtain + intro chime (2.6s).")

    def _stage_preview() -> str:
        ch = s["characters"]
        return render_stage_html(
            [c["name"] for c in ch],
            [c["avatar"] for c in ch],
            s.get("bubbles", ["", "", ""]),
            -1,
        )

    # Main tab: performance shell visible; stage controls stay hidden until curtain finishes.
    yield (
        s,
        render_curtain_html(False),
        gr.update(value=INTRO_CHIME_PATH, autoplay=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        _stage_preview(),
        gr.Tabs(selected=1),
        "\n".join(s["logs"]),
    )
    time.sleep(0.05)
    if _current_epoch() != epoch:
        return

    yield (
        s,
        render_curtain_html(True),
        gr.skip(),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        _stage_preview(),
        gr.Tabs(selected=1),
        "\n".join(s["logs"]),
    )
    time.sleep(2.6)
    if _current_epoch() != epoch:
        return

    s["curtain_raised"] = True
    _log(s, "[Director] Curtain open — Main Stage live (Play unlocked).")
    yield (
        s,
        render_curtain_html(True),
        gr.skip(),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        _stage_preview(),
        gr.Tabs(selected=1),
        "\n".join(s["logs"]),
    )


def reset_all(session: dict[str, Any]) -> tuple[Any, ...]:
    _bump_stream_epoch()
    fresh = initial_session()
    fresh["curtain_raised"] = False
    fresh["logs"] = ["[Director] Full reset — backstage defaults restored."]
    stage = render_stage_html(
        [c["name"] for c in fresh["characters"]],
        [c["avatar"] for c in fresh["characters"]],
        fresh["bubbles"],
        -1,
    )
    return (
        fresh,
        render_curtain_html(False),
        gr.skip(),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        stage,
        gr.Tabs(selected=0),
        "\n".join(fresh["logs"]),
    )


def pause_click(session: dict[str, Any]) -> dict[str, Any]:
    return _ctrl_pause(session, True)


def play_click(session: dict[str, Any]) -> dict[str, Any]:
    """Clear pause so a running ``run_performance`` generator can continue."""
    session["_ctrl"]["pause"] = False
    return session


def stop_click(session: dict[str, Any]) -> dict[str, Any]:
    return _ctrl_stop(session)


def run_performance(
    session: dict[str, Any],
    use_ollama: bool,
    ollama_model: str,
) -> Iterator[tuple[dict[str, Any], str, Any, str]]:
    """
    Multi-agent loop: stream sentences + TTS per actor; interruption engine between clauses.

    Yields: (session, stage_html, line_audio path/update/skip, logs_joined).
    Uses ``gr.skip()`` for audio when nothing new should play so Gradio does not wipe the player.
    """
    if not session.get("curtain_raised"):
        _log(session, "[Router] Curtain not raised — start from Backstage.")
        yield (
            session,
            render_stage_html(
                [c["name"] for c in session["characters"]],
                [c["avatar"] for c in session["characters"]],
                session.get("bubbles", ["", "", ""]),
                -1,
            ),
            gr.skip(),
            "\n".join(session.get("logs", [])),
        )
        return

    session["use_ollama"] = bool(use_ollama)
    session["ollama_model"] = (ollama_model or "").strip() or DEFAULT_OLLAMA_MODEL
    session["dialogue"] = []

    run_epoch = _bump_stream_epoch()
    session["_ctrl"]["stop"] = False
    session["_ctrl"]["pause"] = False
    hf_llm = session.get("hf_llm")
    hf_tts = session.get("hf_tts")
    chars = session["characters"]
    names = [c["name"] for c in chars]
    avatars = [c["avatar"] for c in chars]
    bubbles = session.setdefault("bubbles", ["", "", ""])
    bubbles[:] = ["", "", ""]

    next_speaker = 0
    last_speaker: int | None = None
    max_beats = 36
    beat = 0

    def emit(speaker: int, log_extra: str = "") -> tuple[dict[str, Any], str, Any, str]:
        if log_extra:
            _log(session, log_extra)
        html = render_stage_html(names, avatars, bubbles, speaker)
        return session, html, gr.skip(), "\n".join(session["logs"])

    _log(session, "[Director] Play — performance loop started.")
    if session["use_ollama"]:
        host = os.environ.get("OLLAMA_HOST") or "(default)"
        _log(
            session,
            f"[LLM] Ollama enabled — model `{session['ollama_model']}` host `{host}`.",
        )
    else:
        _log(session, "[LLM] Ollama disabled — rotating canned three-voice lines.")

    yield emit(-1, "")

    tts_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    while beat < max_beats and not session["_ctrl"]["stop"]:
        if _current_epoch() != run_epoch:
            _log(session, "[Director] Epoch changed — performance halted.")
            yield emit(-1, "")
            tts_pool.shutdown(wait=True)
            return

        while session["_ctrl"]["pause"] and not session["_ctrl"]["stop"]:
            yield emit(-1, "")
            time.sleep(0.12)

        if session["_ctrl"]["stop"]:
            break

        speaker = next_speaker
        if last_speaker is not None and last_speaker != speaker:
            bubbles[last_speaker] = ""
        bubbles[speaker] = ""
        last_speaker = speaker

        c = chars[speaker]
        _log(session, f"[Router] Floor → {c['name']} (voice: {c['tone']}).")
        yield emit(speaker, "")

        gen = llm_stream_turn(
            session,
            hf_llm,
            speaker,
            c["name"],
            c["goal"],
            c["secret"],
            c["tone"],
            session["environment"],
            session["topic"],
            list(bubbles),
        )

        interrupted_turn = False
        clause_idx = 0
        turn_clauses: list[str] = []
        for clause in gen:
            clause_idx += 1
            if _current_epoch() != run_epoch or session["_ctrl"]["stop"]:
                interrupted_turn = True
                break

            while session["_ctrl"]["pause"] and not session["_ctrl"]["stop"]:
                yield emit(speaker, "")
                time.sleep(0.1)

            _log(
                session,
                f"[Router] Character {speaker + 1} streaming clause {clause_idx} — {clause[:72]}{'…' if len(clause) > 72 else ''}",
            )

            tts_future = tts_pool.submit(tts_synthesize, hf_tts, clause, c["tone"])

            words = clause.split()
            prefix = ""
            if not words:
                bubbles[speaker] = clause
                yield emit(speaker, "")
            else:
                word_batch = 5
                for wi in range(0, len(words), word_batch):
                    if _current_epoch() != run_epoch or session["_ctrl"]["stop"]:
                        interrupted_turn = True
                        break
                    chunk = words[wi : wi + word_batch]
                    sep = "" if not prefix else " "
                    bubbles[speaker] = prefix + sep + " ".join(chunk)
                    prefix = bubbles[speaker]
                    yield emit(speaker, "")
                    time.sleep(0.02)

            if interrupted_turn:
                tts_future.cancel()
                break

            _log(session, f"[TTS Engine] Readying audio for {c['name']} ({c['tone']}) …")
            try:
                audio_path = tts_future.result(timeout=180)
            except Exception:
                interrupted_turn = True
                break

            yield (
                session,
                render_stage_html(names, avatars, bubbles, speaker),
                gr.update(value=audio_path, autoplay=True),
                "\n".join(session["logs"]),
            )
            if not _sleep_until_playback_end(audio_path, clause, session, run_epoch):
                interrupted_turn = True
                break

            turn_clauses.append(clause)

            if clause_idx >= MIN_CLAUSES_BEFORE_INTERRUPT:
                s0, s1 = _interrupt_scores(speaker, clause, chars, clause_idx, beat)
                others = [j for j in range(3) if j != speaker]
                scores_txt = f"Actor {others[0] + 1}: {s0} | Actor {others[1] + 1}: {s1}"
                _log(
                    session,
                    f"[Interruption Engine] Listener scores — {scores_txt} (threshold {INTERRUPT_SCORE_THRESHOLD}).",
                )

                best_j = others[0] if s0 >= s1 else others[1]
                best_s = max(s0, s1)
                if best_s >= INTERRUPT_SCORE_THRESHOLD:
                    loser = speaker
                    _log(
                        session,
                        f"[Interruption Engine] Character {best_j + 1} interruption score: {best_s}/100 → "
                        f"TRIGGERING INTERRUPT (cuts Actor {loser + 1}).",
                    )
                    yield (
                        session,
                        render_stage_html(names, avatars, bubbles, speaker),
                        gr.update(value=GASP_PATH, autoplay=True),
                        "\n".join(session["logs"]),
                    )
                    _sleep_until_playback_end(GASP_PATH, "", session, run_epoch)
                    bubbles[loser] = ""
                    next_speaker = best_j
                    interrupted_turn = True
                    break

        if turn_clauses:
            session.setdefault("dialogue", []).append(
                {"speaker": speaker, "name": c["name"], "text": " ".join(turn_clauses)}
            )

        if session["_ctrl"]["stop"] or _current_epoch() != run_epoch:
            break

        if not interrupted_turn:
            next_speaker = (speaker + 1) % 3

        beat += 1

    tts_pool.shutdown(wait=True)

    _log(session, "[Director] Performance loop ended (guard limit, Stop, or epoch bump).")
    yield emit(-1, "")


def attach_hf_clients(session: dict[str, Any], llm_client: Any, tts_client: Any) -> dict[str, Any]:
    """Optional: stash HF ``InferenceClient`` instances on the session before ``Play``."""
    session["hf_llm"] = llm_client
    session["hf_tts"] = tts_client
    return session


def build_app() -> gr.Blocks:
    avatars_choices = ["🎭", "🦊", "🤖", "👑", "🐸", "🦄", "🐙", "🦉"]
    tone_choices = list(VOICE_PRESETS.keys())
    env_choices = ["Garden", "Hospital", "Living Room", "Park"]

    with gr.Blocks(title="AI Puppet Theater With Tiny Actors") as demo:
        session_state = gr.State(initial_session())

        gr.Markdown(
            "## AI Puppet Theater — *Tiny Actors, Human-Like Streaming*\n"
            "Configure **Backstage**, then **Raise Curtain** for the curtain animation plus intro chime. "
            "On **Main Stage**, set **Ollama** (or use offline lines), then **Play** for LLM dialogue + TTS placeholders. "
            "**Pause** / **Resume** / **Stop** control the loop; **Reset** clears scene state."
        )

        tabs = gr.Tabs(selected=0)
        with tabs:
            with gr.Tab("Backstage (Setup)"):
                gr.Markdown("### Scene setup")
                env = gr.Dropdown(env_choices, value="Garden", label="Environment")
                topic = gr.Textbox(label="Global topic", lines=2, value="A missing invitation")

                gr.Markdown("### Character configuration — three columns")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Actor 1**")
                        a1_avatar = gr.Dropdown(avatars_choices, value="🎭", label="Avatar")
                        a1_goal = gr.Textbox(label="Goal", value="Host the perfect garden party")
                        a1_secret = gr.Textbox(label="Secret", value="Forgot to send half the invites")
                        a1_tone = gr.Dropdown(tone_choices, value=tone_choices[0], label="Tone / style")
                    with gr.Column():
                        gr.Markdown("**Actor 2**")
                        a2_avatar = gr.Dropdown(avatars_choices, value="🦊", label="Avatar")
                        a2_goal = gr.Textbox(label="Goal", value="Keep everyone calm")
                        a2_secret = gr.Textbox(label="Secret", value="Knows the caterer canceled")
                        a2_tone = gr.Dropdown(tone_choices, value=tone_choices[1], label="Tone / style")
                    with gr.Column():
                        gr.Markdown("**Actor 3**")
                        a3_avatar = gr.Dropdown(avatars_choices, value="🤖", label="Avatar")
                        a3_goal = gr.Textbox(label="Goal", value="Log outcomes objectively")
                        a3_secret = gr.Textbox(label="Secret", value="Was told to spin the report")
                        a3_tone = gr.Dropdown(tone_choices, value=tone_choices[2], label="Tone / style")

                with gr.Row():
                    raise_btn = gr.Button("Raise Curtain", variant="primary")
                    reset_backstage = gr.Button("Reset scene", variant="secondary")

            with gr.Tab("Main Stage (Performance)"):
                locked_md = gr.Markdown(
                    "🔒 **The curtain is down.** Open **Backstage** and press **Raise Curtain** to unlock "
                    "the intro animation, chime, and controls.",
                )
                perf_column = gr.Column(visible=False)
                with perf_column:
                    curtain_html = gr.HTML(value=render_curtain_html(False))
                    intro_audio = gr.Audio(
                        label="Theater intro / live line (autoplay)",
                        type="filepath",
                        format="wav",
                        autoplay=True,
                        interactive=False,
                    )
                    stage_inner = gr.Column(visible=False)
                    with stage_inner:
                        stage_html = gr.HTML(
                            value=render_stage_html(
                                ["Actor 1", "Actor 2", "Actor 3"],
                                ["🎭", "🦊", "🤖"],
                                ["", "", ""],
                                -1,
                            ),
                        )
                        gr.Markdown(
                            "**Dialogue** is generated with **local Ollama** when enabled (see Day 2/4). "
                            "Run `ollama serve` and `ollama pull <model>`; uncheck for short offline lines."
                        )
                        with gr.Row():
                            use_ollama_cb = gr.Checkbox(value=True, label="Use Ollama (local LLM)")
                            ollama_model_tb = gr.Textbox(
                                label="Ollama model",
                                value=os.environ.get("OLLAMA_MODEL") or "llama3.2",
                            )
                        line_audio = gr.Audio(
                            label="Live line (TTS — edge-tts MP3 per clause; gasp is WAV)",
                            type="filepath",
                            autoplay=True,
                            interactive=False,
                        )
                        with gr.Row():
                            play_btn = gr.Button("Play", variant="primary")
                            resume_btn = gr.Button("Resume")
                            pause_btn = gr.Button("Pause")
                            stop_btn = gr.Button("Stop", variant="stop")
                        with gr.Accordion("Behind-the-Scenes Logs", open=True):
                            logs_box = gr.Code(
                                label=None,
                                language=None,
                                lines=20,
                                interactive=False,
                                value="",
                            )

        demo.queue(default_concurrency_limit=20)

        pack_inputs = [
            env,
            topic,
            a1_avatar,
            a1_goal,
            a1_secret,
            a1_tone,
            a2_avatar,
            a2_goal,
            a2_secret,
            a2_tone,
            a3_avatar,
            a3_goal,
            a3_secret,
            a3_tone,
            session_state,
        ]

        curtain_outputs = [
            session_state,
            curtain_html,
            intro_audio,
            locked_md,
            perf_column,
            stage_inner,
            stage_html,
            tabs,
            logs_box,
        ]

        def _pack_and_raise(*args: Any) -> Any:
            *ui, sess = args
            s = pack_session_from_ui(*ui, sess)
            yield from raise_curtain_flow(s)

        raise_btn.click(
            _pack_and_raise,
            inputs=pack_inputs,
            outputs=curtain_outputs,
        )

        reset_backstage.click(
            reset_all,
            inputs=[session_state],
            outputs=curtain_outputs,
        )

        play_btn.click(
            run_performance,
            inputs=[session_state, use_ollama_cb, ollama_model_tb],
            outputs=[session_state, stage_html, line_audio, logs_box],
        )

        resume_btn.click(play_click, inputs=[session_state], outputs=[session_state])
        pause_btn.click(pause_click, inputs=[session_state], outputs=[session_state])
        stop_btn.click(stop_click, inputs=[session_state], outputs=[session_state])

    return demo


def main() -> None:
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True,
        theme=THEATER_THEME,
        css=THEATER_CSS,
    )


if __name__ == "__main__":
    main()
