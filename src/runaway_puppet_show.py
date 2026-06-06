"""
The Runaway Puppet Show — Thousand Token Wood (Gradio).

AI-native puppet theater: live unscripted beats, audience throws / reality rules,
chaos meter, tiny per-puppet memories, optional HF image gen + optional Ollama/HF text.

Run locally:
  ollama serve && ollama pull llama3.2
  uv run python src/runaway_puppet_show.py

Hugging Face Space: set HF_TOKEN (secret) for Inference API; optional OLLAMA_* if you
proxy Ollama. PORT is respected automatically.
"""

from __future__ import annotations

import base64
import copy
import html
import json
import os
import random
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from typing import Any

import gradio as gr
from ollama import Client as OllamaClient
from pathlib import Path

from pydantic import BaseModel, Field

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover
    InferenceClient = None  # type: ignore[misc, assignment]

# --- stream epoch (stop in-flight show) ---
_epoch_lock = threading.Lock()
_stream_epoch = [0]


def _bump_epoch() -> int:
    with _epoch_lock:
        _stream_epoch[0] += 1
        return _stream_epoch[0]


def _current_epoch() -> int:
    return _stream_epoch[0]


DEFAULT_OLLAMA = os.environ.get("OLLAMA_MODEL", "llama3.2")
HF_LLM = os.environ.get("HF_LLM_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct:hf-infer")
HF_IMAGE = os.environ.get("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")


class BeatLine(BaseModel):
    who: str = Field(description="Speaker name exactly as in roster, or NARRATOR")
    text: str = Field(description="One short theatrical line, no stage directions")


class BeatResponse(BaseModel):
    lines: list[BeatLine] = Field(min_length=2, max_length=12)


class NewPuppetSpec(BaseModel):
    name: str
    personality: str
    goal: str
    fear: str
    secret: str
    avatar_prompt: str


def _chaos_band(c: int) -> str:
    if c < 20:
        return "0-20 NORMAL: coherent fairytale puppet plot, gentle jokes."
    if c < 50:
        return "20-50 IMPROV: wilder tangents, props appear, characters escalate quirks."
    if c < 80:
        return "50-80 META-TENSION: characters bicker with the narrator's framing."
    if c < 100:
        return "80-99 FOURTH WALL: characters suspect something is 'off' about tonight."
    return "100 CHAOS MAX: puppets know they are AI puppets; reference past nights below."


def default_puppets() -> list[dict[str, Any]]:
    return [
        {
            "name": "Sock Dragon",
            "personality": "Grandiose, dramatic, oddly sincere",
            "goal": "Convince everyone he is a real dragon",
            "fear": "Being called a sock",
            "secret": "Actually knows he is a sock",
            "avatar": "🧦🐉",
            "avatar_prompt": "paper cutout sock dragon puppet, storybook watercolor, wood stage",
            "tone": "Deep/Dramatic",
            "memory": deque(maxlen=4),
        },
        {
            "name": "Sir Croaksalot",
            "personality": "Ambitious frog knight, puffed chest, squeaks when nervous",
            "goal": "Become king",
            "fear": "Public speaking",
            "secret": "Already stole the crown",
            "avatar": "🐸👑",
            "avatar_prompt": "frog knight puppet with tiny crown, cutout paper, whimsical",
            "tone": "High-pitched/Nervous",
            "memory": deque(maxlen=4),
        },
        {
            "name": "Gerald the Cactus Accountant",
            "personality": "Dry, numbers-first, secretly sentimental",
            "goal": "Balance the kingdom budget",
            "fear": "Unexpected emotions",
            "secret": "Writes poetry",
            "avatar": "🌵📒",
            "avatar_prompt": "cactus accountant puppet with ledger, soft storybook ink",
            "tone": "Robotic/Monotone",
            "memory": deque(maxlen=4),
        },
        {
            "name": "Mistleaf the Fox",
            "personality": "Clever, teasing, pretends to know more than she does",
            "goal": "Win every argument without trying",
            "fear": "Being ignored",
            "secret": "Cannot read maps",
            "avatar": "🦊",
            "avatar_prompt": "fox puppet paper cutout, lantern light, cozy wood theater",
            "tone": "High-pitched/Nervous",
            "memory": deque(maxlen=4),
        },
    ]


def fresh_session() -> dict[str, Any]:
    pups = default_puppets()
    return {
        "show_on": False,
        "curtain": False,
        "chaos": 0,
        "transcript": [],
        "puppets": pups,
        "reality_rules": [],
        "audience_queue": [],
        "past_show_summaries": [],
        "beat_count": 0,
        "stage_bg": "wood_paper",
        "puppet_images": {},  # name -> filepath optional
        "last_error": "",
    }


def _fix_session_types(s: dict[str, Any]) -> dict[str, Any]:
    """Ensure list/deque types after deepcopy/json."""
    if not isinstance(s.get("transcript"), list):
        s["transcript"] = []
    if not isinstance(s.get("reality_rules"), list):
        s["reality_rules"] = []
    if not isinstance(s.get("audience_queue"), list):
        s["audience_queue"] = []
    if not isinstance(s.get("past_show_summaries"), list):
        s["past_show_summaries"] = []
    for p in s.get("puppets", []):
        if not isinstance(p.get("memory"), deque):
            m = p.get("memory") or []
            d: deque[str] = deque(maxlen=4)
            for x in list(m)[-4:]:
                d.append(str(x))
            p["memory"] = d
    return s


def fresh_session_typed() -> dict[str, Any]:
    s = fresh_session()
    s["transcript"] = []
    s["reality_rules"] = []
    s["audience_queue"] = []
    s["past_show_summaries"] = []
    return s


def _ollama_ok(host: str | None) -> bool:
    try:
        OllamaClient(host=host).list()
        return True
    except Exception:
        return False


def _hf_client() -> Any | None:
    if InferenceClient is None:
        return None
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        return None
    return InferenceClient(token=tok)


def llm_beat_json(
    *,
    use_ollama: bool,
    ollama_model: str,
    system: str,
    user: str,
) -> BeatResponse | None:
    host = os.environ.get("OLLAMA_HOST")
    schema = BeatResponse.model_json_schema()
    if use_ollama and _ollama_ok(host):
        try:
            raw = OllamaClient(host=host).chat(
                model=(ollama_model or DEFAULT_OLLAMA).strip() or DEFAULT_OLLAMA,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + "\n\nReply with JSON only matching schema: {lines:[{who,text},...]}"},
                ],
                format=schema,
                options={"temperature": 0.9, "num_predict": 700},
                stream=False,
            )
            content = (raw.message.content or "").strip()
            return BeatResponse.model_validate_json(content)
        except Exception:
            return None

    cli = _hf_client()
    if cli is not None:
        try:
            prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
            out = cli.text_generation(
                prompt,
                model=HF_LLM,
                max_new_tokens=512,
                temperature=0.9,
            )
            m = re.search(r"\{[\s\S]*\}", out)
            if m:
                return BeatResponse.model_validate_json(m.group(0))
        except Exception:
            return None
    return None


def mock_beat(puppets: list[dict[str, Any]], chaos: int, thrown: str | None, rules: list[str]) -> BeatResponse:
    names = [p["name"] for p in puppets]
    lines: list[BeatLine] = [
        BeatLine(
            who="NARRATOR",
            text="The curtain trembles — something impossible leans in from the wings.",
        )
    ]
    if thrown:
        lines.append(BeatLine(who=random.choice(names), text=f"The audience flings a {thrown}! We must improvise!"))
    for p in puppets[:2]:
        snippet = (p["goal"] or "dream")[:40]
        lines.append(BeatLine(who=p["name"], text=f"I swear on my buttons: {snippet} matters tonight!"))
    if chaos >= 80:
        lines.append(BeatLine(who="NARRATOR", text="(whispered) Even I am not sure who wrote this page."))
    if rules:
        lines.append(BeatLine(who="NARRATOR", text=f"New law of the wood: {'; '.join(rules[-2:])}"))
    return BeatResponse(lines=lines[:8])


def generate_new_puppet(user_idea: str, *, use_ollama: bool, ollama_model: str) -> NewPuppetSpec:
    system = (
        "You invent ONE puppet for Thousand Token Wood. JSON only, keys: "
        "name, personality, goal, fear, secret, avatar_prompt (visual prompt for a square storybook puppet portrait)."
    )
    user = f"Audience request: {user_idea.strip()[:400]}"
    host = os.environ.get("OLLAMA_HOST")
    schema = NewPuppetSpec.model_json_schema()
    if use_ollama and _ollama_ok(host):
        try:
            raw = OllamaClient(host=host).chat(
                model=(ollama_model or DEFAULT_OLLAMA).strip() or DEFAULT_OLLAMA,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=schema,
                options={"temperature": 0.85, "num_predict": 400},
                stream=False,
            )
            return NewPuppetSpec.model_validate_json((raw.message.content or "").strip())
        except Exception:
            pass
    cli = _hf_client()
    if cli is not None:
        try:
            out = cli.text_generation(
                f"{system}\n\n{user}\nJSON:",
                model=HF_LLM,
                max_new_tokens=350,
                temperature=0.85,
            )
            m = re.search(r"\{[\s\S]*\}", out)
            if m:
                return NewPuppetSpec.model_validate_json(m.group(0))
        except Exception:
            pass
    # fallback puppet
    base = user_idea.strip()[:32] or "Mystery"
    return NewPuppetSpec(
        name=f"{base} Puppet",
        personality="Shy, hopeful, slightly chaotic",
        goal="Belong in the cast",
        fear="Spotlights",
        secret="Was sewn from spare thread",
        avatar_prompt=f"paper puppet theater portrait of {base}, watercolor, cutout",
    )


def maybe_generate_portrait(spec: NewPuppetSpec) -> str | None:
    cli = _hf_client()
    if cli is None:
        return None
    try:
        bio = cli.text_to_image(
            spec.avatar_prompt + ", square, storybook puppet theater, soft paper texture, transparent background preferred",
            model=HF_IMAGE,
        )
        if bio is None:
            return None
        d = path_for_session_image(spec.name)
        if isinstance(bio, bytes):
            d.write_bytes(bio)
        elif hasattr(bio, "save"):
            bio.save(str(d))  # type: ignore[union-attr]
        else:
            d.write_bytes(bio.read())  # type: ignore[attr-defined]
        return str(d)
    except Exception:
        return None


def path_for_session_image(name: str) -> Path:
    root = Path(os.environ.get("TMPDIR", "/tmp")) / "runaway_puppet_imgs"
    root.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)[:40]
    return root / f"{safe}_{uuid.uuid4().hex[:8]}.png"


def _image_file_data_uri(path: str) -> str | None:
    try:
        raw = Path(path).read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def summarize_show_for_memory(transcript: list[dict[str, str]], *, use_ollama: bool, ollama_model: str) -> str:
    """One line 'what happened last night' for chaos-100 callbacks."""
    tail = transcript[-16:]
    blob = " | ".join(f"{t['who']}: {t['text'][:80]}" for t in tail)
    if not blob:
        return "Someone threw vegetables at the moon."
    host = os.environ.get("OLLAMA_HOST")
    if use_ollama and _ollama_ok(host):
        try:
            raw = OllamaClient(host=host).chat(
                model=(ollama_model or DEFAULT_OLLAMA).strip() or DEFAULT_OLLAMA,
                messages=[
                    {
                        "role": "system",
                        "content": "One funny sentence, past tense, puppets recall a previous show. No quotes.",
                    },
                    {"role": "user", "content": f"Transcript tail:\n{blob}"},
                ],
                options={"temperature": 0.8, "num_predict": 80},
                stream=False,
            )
            return (raw.message.content or "").strip().split("\n")[0][:220]
        except Exception:
            pass
    cli = _hf_client()
    if cli is not None:
        try:
            out = cli.text_generation(
                f"One funny sentence past tense summarizing puppet chaos:\n{blob}\nSentence:",
                model=HF_LLM,
                max_new_tokens=60,
                temperature=0.7,
            )
            return out.strip().split("\n")[0][:220]
        except Exception:
            pass
    return random.choice(
        [
            "Yesterday somebody turned us all into potatoes.",
            "Last week the narrator lost a fight to a rubber duck.",
            "Opening night, gravity went sideways and the budget caught fire.",
        ]
    )


def build_beat_user_prompt(session: dict[str, Any]) -> str:
    pups = session["puppets"]
    roster = []
    for p in pups:
        mem = " | ".join(p["memory"]) if p["memory"] else "(nothing — fuzzy head)"
        roster.append(
            f"- {p['name']}: personality={p['personality']}; goal={p['goal']}; fear={p['fear']}; "
            f"secret={p['secret']} — THEY ONLY REMEMBER: {mem}"
        )
    rules = "; ".join(session["reality_rules"]) if session["reality_rules"] else "(none)"
    thrown = session.get("_pending_throw")
    past = session.get("past_show_summaries") or []
    past_txt = "\n".join(f"- {x}" for x in past[-6:]) if past else "(no prior nights yet)"
    transcript = session.get("transcript") or []
    last = "\n".join(f"{t['who']}: {t['text']}" for t in transcript[-14:])
    chaos = int(session.get("chaos", 0))
    band = _chaos_band(chaos)
    extra = ""
    if chaos >= 100:
        extra = (
            "\nCHAOS 100 MODE: puppets may joke that they are AI / generated. "
            "They MUST reference at least one 'past night' rumor from this list literally:\n"
            f"{past_txt}"
        )
    return (
        f"Setting: Thousand Token Wood — tiny puppet theater inside an old storybook.\n"
        f"Active reality rules: {rules}\n"
        f"Audience just threw: {thrown or 'nothing'}\n"
        f"Chaos now: {chaos}. Band: {band}\n"
        f"{extra}\n"
        f"Recent transcript:\n{last or '(curtain just rose)'}\n\n"
        f"Roster:\n" + "\n".join(roster) + "\n\n"
        "Write the next 3–8 short spoken lines as JSON.lines. "
        "Include NARRATOR at least once. Lines must be witty, theatrical, under ~18 words. "
        "No markdown, no asterisks. who must match a roster name or NARRATOR."
    )


NARRATOR_SYSTEM = """You are the combined narrator + cast director for a puppet show.
Combine character voices, keep continuity with the transcript, lean into comedy from conflicts.
Never long paragraphs. Puppet lines feel like stage dialogue."""


def apply_audience_queue(session: dict[str, Any]) -> None:
    q = session["audience_queue"]
    session["_pending_throw"] = None
    while q:
        item = q.pop(0)
        kind = item.get("kind")
        if kind == "throw":
            session["_pending_throw"] = item.get("payload", "mystery object")
            session["chaos"] = min(100, int(session["chaos"]) + 8)
        elif kind == "reality":
            rule = (item.get("payload") or "").strip()
            if rule:
                session["reality_rules"].append(rule)
            session["chaos"] = min(100, int(session["chaos"]) + 12)


def render_stage_html(session: dict[str, Any]) -> str:
    pups = session["puppets"]
    imgs: dict[str, Any] = session.get("puppet_images") or {}
    cards = []
    for p in pups:
        name = html.escape(p["name"])
        av = html.escape(p.get("avatar", "🎭"))
        path = imgs.get(p["name"])
        data_uri = _image_file_data_uri(path) if path and os.path.isfile(path) else None
        if data_uri:
            inner = (
                f'<img src="{data_uri}" alt="" '
                'style="width:72px;height:72px;object-fit:cover;border-radius:12px;"/>'
            )
        else:
            inner = f'<span style="font-size:2.4rem;">{av}</span>'
        cards.append(
            f'<div class="puppet-card"><div class="puppet-avatar">{inner}</div>'
            f'<div class="puppet-name">{name}</div></div>'
        )
    chaos = int(session.get("chaos", 0))
    return f"""
<div class="stage-frame">
  <div class="proscenium">Thousand Token Wood</div>
  <div class="chaos-meter"><span>Chaos</span><div class="chaos-bar"><i style="width:{chaos}%"></i></div><span>{chaos}</span></div>
  <div class="puppet-row">{"".join(cards)}</div>
</div>
"""


def transcript_to_chat(transcript: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for t in transcript:
        role = "assistant"
        who = t.get("who", "?")
        txt = t.get("text", "")
        out.append({"role": role, "content": f"**{who}:** {txt}"})
    return out


def run_show(
    session: dict[str, Any],
    use_ollama: bool,
    ollama_model: str,
    max_beats: int,
) -> Iterator[tuple[Any, ...]]:
    session = _fix_session_types(copy.deepcopy(session))
    session["show_on"] = True
    session["curtain"] = True
    epoch = _bump_epoch()
    yield (
        session,
        render_stage_html(session),
        transcript_to_chat(session.get("transcript", [])),
        chaos_label(session),
        secrets_markdown(session),
        replay_json(session),
        gr.update(interactive=False),
        "",
    )

    beat = 0
    while beat < int(max_beats) and _current_epoch() == epoch:
        apply_audience_queue(session)
        user = build_beat_user_prompt(session)
        thrown = session.pop("_pending_throw", None)
        if thrown:
            session["_last_throw"] = thrown

        br = llm_beat_json(use_ollama=use_ollama, ollama_model=ollama_model, system=NARRATOR_SYSTEM, user=user)
        if br is None:
            br = mock_beat(session["puppets"], int(session["chaos"]), thrown, session["reality_rules"])

        for line in br.lines:
            who = line.who.strip()
            text = line.text.strip()
            if not text:
                continue
            session.setdefault("transcript", []).append({"who": who, "text": text})
            mem_bit = f"{who}: {text[:100]}"
            for p in session["puppets"]:
                p["memory"].append(mem_bit)
            yield (
                session,
                render_stage_html(session),
                transcript_to_chat(session["transcript"]),
                chaos_label(session),
                secrets_markdown(session),
                replay_json(session),
                gr.update(interactive=False),
                "",
            )
            time.sleep(0.04)

        session["beat_count"] = int(session.get("beat_count", 0)) + 1
        beat += 1
        yield (
            session,
            render_stage_html(session),
            transcript_to_chat(session["transcript"]),
            chaos_label(session),
            secrets_markdown(session),
            replay_json(session),
            gr.update(interactive=False),
            "",
        )

    session["show_on"] = False
    yield (
        session,
        render_stage_html(session),
        transcript_to_chat(session.get("transcript", [])),
        chaos_label(session),
        secrets_markdown(session),
        replay_json(session),
        gr.update(interactive=True),
        "Show loop ended — throw more objects or Start Show again.",
    )


def chaos_label(session: dict[str, Any]) -> str:
    c = int(session.get("chaos", 0))
    band = _chaos_band(c).split(":")[0]
    return f"**{c}/100** — {band}"


def secrets_markdown(session: dict[str, Any]) -> str:
    lines = ["### Spoilers — cast secrets\n"]
    for p in session.get("puppets", []):
        lines.append(
            f"- **{p['name']}:** _{html.escape(str(p.get('secret','')))}_ "
            f"(fear: {html.escape(str(p.get('fear','')))})"
        )
    return "\n".join(lines)


def replay_json(session: dict[str, Any]) -> str:
    export = {
        "title": "The Runaway Puppet Show",
        "chaos": session.get("chaos"),
        "reality_rules": session.get("reality_rules"),
        "transcript": session.get("transcript"),
        "past_show_summaries": session.get("past_show_summaries"),
    }
    return json.dumps(export, indent=2, ensure_ascii=False)


def queue_throw(session: dict[str, Any], obj: str) -> dict[str, Any]:
    session = _fix_session_types(copy.deepcopy(session))
    session.setdefault("audience_queue", []).append({"kind": "throw", "payload": (obj or "potato").strip()})
    return session


def queue_reality(session: dict[str, Any], rule: str) -> dict[str, Any]:
    session = _fix_session_types(copy.deepcopy(session))
    session.setdefault("audience_queue", []).append({"kind": "reality", "payload": (rule or "").strip()})
    return session


def add_character_flow(
    session: dict[str, Any],
    idea: str,
    use_ollama: bool,
    ollama_model: str,
) -> tuple[dict[str, Any], str, str]:
    session = _fix_session_types(copy.deepcopy(session))
    spec = generate_new_puppet(idea, use_ollama=use_ollama, ollama_model=ollama_model)
    pup = {
        "name": spec.name,
        "personality": spec.personality,
        "goal": spec.goal,
        "fear": spec.fear,
        "secret": spec.secret,
        "avatar": "✨",
        "avatar_prompt": spec.avatar_prompt,
        "tone": "Robotic/Monotone",
        "memory": deque(maxlen=4),
    }
    session.setdefault("puppets", []).append(pup)
    session["chaos"] = min(100, int(session.get("chaos", 0)) + 18)
    path = maybe_generate_portrait(spec)
    if path:
        session.setdefault("puppet_images", {})[spec.name] = path
    session.setdefault("audience_queue", []).append(
        {"kind": "reality", "payload": f"A new puppet joins: {spec.name} ({spec.personality})."}
    )
    blurb = (
        f"Added **{spec.name}** — {spec.personality}. Goal: {spec.goal}. "
        f"(Portrait: {'generated' if path else 'emoji placeholder'})"
    )
    return session, render_stage_html(session), blurb


def end_show_remember(session: dict[str, Any], use_ollama: bool, ollama_model: str) -> tuple[dict[str, Any], str]:
    session = _fix_session_types(copy.deepcopy(session))
    summ = summarize_show_for_memory(session.get("transcript", []), use_ollama=use_ollama, ollama_model=ollama_model)
    session.setdefault("past_show_summaries", []).append(summ)
    session["show_on"] = False
    _bump_epoch()
    return session, f"Remembered for future chaos: _{summ}_"


def reset_all() -> dict[str, Any]:
    _bump_epoch()
    return fresh_session_typed()


def stop_show() -> None:
    _bump_epoch()


SHOW_CSS = """
.gradio-container {
  background: linear-gradient(165deg, #3d2918 0%, #1e120c 45%, #0c0704 100%) !important;
}
.runaway-hero {
  text-align: center;
  padding: 1rem 1rem 1.25rem;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,245,220,0.12), rgba(40,24,12,0.35));
  border: 1px solid rgba(210, 170, 120, 0.35);
  box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}
.runaway-hero h1 { font-family: Georgia, 'Times New Roman', serif; color: #fff4e0 !important; }
.runaway-hero .tag { color: #e8d4bc !important; font-style: italic; }
.stage-frame {
  border-radius: 16px;
  padding: 14px 12px 18px;
  background:
    repeating-linear-gradient(90deg, rgba(60,40,20,0.15) 0 2px, transparent 2px 14px),
    linear-gradient(180deg, #5a3a22, #2a1810);
  border: 3px solid #7a5230;
  box-shadow: inset 0 0 0 2px rgba(255,220,180,0.08), 0 18px 50px rgba(0,0,0,0.45);
}
.proscenium {
  text-align: center;
  font-family: Georgia, serif;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-size: 0.78rem;
  color: #f2e2c8;
  opacity: 0.9;
  margin-bottom: 8px;
}
.chaos-meter {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #fdebd2;
  font-size: 0.85rem;
  margin-bottom: 12px;
}
.chaos-bar {
  flex: 1;
  height: 10px;
  border-radius: 999px;
  background: rgba(0,0,0,0.35);
  overflow: hidden;
  border: 1px solid rgba(255,200,140,0.25);
}
.chaos-bar > i {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #f4c542, #e0583a, #7c3aed);
}
.puppet-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 14px;
}
.puppet-card {
  min-width: 110px;
  text-align: center;
  padding: 8px 8px 10px;
  border-radius: 14px;
  background: rgba(255, 248, 235, 0.06);
  border: 1px solid rgba(255, 220, 180, 0.15);
}
.puppet-name { color: #fdebd2; font-size: 0.78rem; margin-top: 6px; }
.puppet-avatar { filter: drop-shadow(0 6px 10px rgba(0,0,0,0.45)); }
"""


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="stone",
        font=[gr.themes.GoogleFont("Nunito"), "ui-sans-serif", "system-ui"],
    ).set(
        body_background_fill_dark="#0f0906",
        block_background_fill_dark="#1a100a",
        block_border_width="1px",
        block_radius="lg",
    )

    with gr.Blocks(title="The Runaway Puppet Show") as demo:
        state = gr.State(fresh_session_typed())

        gr.Markdown(
            "# The Runaway Puppet Show\n"
            '<p class="tag">A puppet theater where the puppets write the play — <em>Thousand Token Wood</em>.</p>',
            elem_classes=["runaway-hero"],
        )

        with gr.Row():
            stage = gr.HTML(value=render_stage_html(fresh_session_typed()))
        chaos_md = gr.Markdown(value=chaos_label(fresh_session_typed()))

        chat = gr.Chatbot(label="Tonight's transcript", height=360)
        status = gr.Markdown("")

        with gr.Accordion("Audience powers", open=True):
            with gr.Row():
                throw = gr.Dropdown(
                    choices=[
                        "potato",
                        "rubber duck",
                        "moon rock",
                        "haunted teacup",
                        "banana sword",
                    ],
                    value="rubber duck",
                    label="Throw object",
                )
                throw_btn = gr.Button("Throw!", variant="primary")
            reality = gr.Textbox(
                label="Change reality",
                placeholder="Everyone speaks like pirates…",
                lines=2,
            )
            reality_btn = gr.Button("Apply rule")

        with gr.Row():
            add_puppet_idea = gr.Textbox(
                label="Add character",
                placeholder="Add a depressed cactus accountant…",
                lines=2,
            )
            add_btn = gr.Button("Summon puppet")

        with gr.Row():
            use_ollama = gr.Checkbox(value=True, label="Use Ollama (local JSON beats)")
            ollama_model = gr.Textbox(label="Ollama model", value=os.environ.get("OLLAMA_MODEL", "llama3.2"))
            max_beats = gr.Slider(4, 32, value=12, step=1, label="Beats per Start Show")

        with gr.Row():
            start = gr.Button("Start Show", variant="primary")
            stop = gr.Button("Stop", variant="stop")
            remember = gr.Button("End show & remember (WOW setup)")
            reset = gr.Button("Reset theater")

        with gr.Accordion("Director's secret file", open=False):
            secrets = gr.Markdown(secrets_markdown(fresh_session_typed()))

        with gr.Accordion("Shareable replay (JSON)", open=False):
            replay = gr.Code(label="Export", language="json", lines=14)

        demo.queue(default_concurrency_limit=12)

        def _ui_pack(sess: dict[str, Any]) -> tuple[Any, ...]:
            s = _fix_session_types(copy.deepcopy(sess))
            return (
                s,
                render_stage_html(s),
                transcript_to_chat(s.get("transcript", [])),
                chaos_label(s),
                secrets_markdown(s),
                replay_json(s),
            )

        def on_throw(s: dict[str, Any], obj: str) -> tuple[Any, ...]:
            s2 = queue_throw(s, obj)
            return (*_ui_pack(s2), gr.update(), "")

        def on_reality(s: dict[str, Any], rule: str) -> tuple[Any, ...]:
            s2 = queue_reality(s, rule)
            return (*_ui_pack(s2), gr.update(), "")

        def on_add(s: dict[str, Any], idea: str, uo: bool, om: str) -> tuple[Any, ...]:
            s2, st_html, msg = add_character_flow(s, idea, uo, om)
            s2 = _fix_session_types(s2)
            return (
                s2,
                st_html,
                transcript_to_chat(s2.get("transcript", [])),
                chaos_label(s2),
                secrets_markdown(s2),
                replay_json(s2),
                gr.update(),
                msg,
            )

        def on_stop(s: dict[str, Any]) -> tuple[Any, ...]:
            stop_show()
            s2 = _fix_session_types(copy.deepcopy(s))
            s2["show_on"] = False
            return (
                *(_ui_pack(s2)),
                gr.update(interactive=True),
                "Cut! The wood goes quiet.",
            )

        throw_btn.click(
            fn=on_throw,
            inputs=[state, throw],
            outputs=[state, stage, chat, chaos_md, secrets, replay, start, status],
        )

        reality_btn.click(
            fn=on_reality,
            inputs=[state, reality],
            outputs=[state, stage, chat, chaos_md, secrets, replay, start, status],
        )

        add_btn.click(
            fn=on_add,
            inputs=[state, add_puppet_idea, use_ollama, ollama_model],
            outputs=[state, stage, chat, chaos_md, secrets, replay, start, status],
        )

        start.click(
            fn=run_show,
            inputs=[state, use_ollama, ollama_model, max_beats],
            outputs=[state, stage, chat, chaos_md, secrets, replay, start, status],
        )

        stop.click(
            fn=on_stop,
            inputs=[state],
            outputs=[state, stage, chat, chaos_md, secrets, replay, start, status],
        )

        remember.click(
            fn=lambda s, uo, om: end_show_remember(s, uo, om),
            inputs=[state, use_ollama, ollama_model],
            outputs=[state, status],
        ).then(
            fn=lambda s: (
                s,
                render_stage_html(s),
                transcript_to_chat(s.get("transcript", [])),
                chaos_label(s),
                secrets_markdown(s),
                replay_json(s),
            ),
            inputs=[state],
            outputs=[state, stage, chat, chaos_md, secrets, replay],
        )

        reset.click(
            fn=reset_all,
            outputs=[state],
        ).then(
            fn=lambda s: (
                s,
                render_stage_html(s),
                transcript_to_chat(s.get("transcript", [])),
                chaos_label(s),
                secrets_markdown(s),
                replay_json(s),
                "",
                gr.update(interactive=True),
            ),
            inputs=[state],
            outputs=[state, stage, chat, chaos_md, secrets, replay, status, start],
        )

    return demo, theme


def main() -> None:
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7863")))
    host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    demo, theme = build_app()
    demo.launch(
        server_name=host,
        server_port=port,
        show_error=True,
        theme=theme,
        css=SHOW_CSS,
    )


if __name__ == "__main__":
    main()
