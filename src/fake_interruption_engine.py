"""
Day 1: Fake Interruption Engine — concurrent actor streams + context isolation.

Actor A appends one word per second into stream state held in a ContextVar.
Listener B polls every 2 seconds; on trigger word "apple", B sets an Event,
cancels A, and streams its own words into the same scene-local buffer.
Two scenes run under copy_context() so tenant state does not bleed.

Implementation note: asyncio gives each Task a copy of the context; re-binding a
ContextVar to a new string in one actor would not update a sibling task. The
buffer therefore uses a ContextVar to a mutable BufferState so co-tasks in one
scene share one stream while different copy_context() scenes use different
BufferState instances.
"""

from __future__ import annotations

import asyncio
import contextvars
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field

TRIGGER = "apple"


@dataclass
class BufferState:
    """
    Mutable stream buffer shared by tasks in one scene.

    asyncio copies context into each new Task; sibling tasks still share the same
    *objects* bound in the parent context until a ContextVar is re-bound. In-place
    mutation on this object lets Actor A and Listener B see one buffer without
    cross-scene leakage (each scene gets its own BufferState via copy_context()).
    """

    _parts: list[str] = field(default_factory=list)

    def append(self, word: str) -> str:
        self._parts.append(f"{word} ")
        return self.text()

    def text(self) -> str:
        return "".join(self._parts)


# Per-scene state (isolated when each scene runs inside its own copied context).
buffer_var: contextvars.ContextVar[BufferState | None] = contextvars.ContextVar(
    "buffer",
    default=None,
)
speaker_var: contextvars.ContextVar[str] = contextvars.ContextVar("speaker", default="A")


def get_buffer() -> str:
    buf = buffer_var.get()
    if buf is None:
        return ""
    return buf.text()


def append_word(word: str) -> str:
    buf = buffer_var.get()
    if buf is None:
        msg = "buffer_var must be set to a BufferState in run_scene before streaming"
        raise RuntimeError(msg)
    return buf.append(word)


def set_speaker(name: str) -> None:
    speaker_var.set(name)


def log(scene_id: str, message: str) -> None:
    print(f"[{scene_id}] {message}", flush=True)


async def actor_a(scene_id: str, words: Iterable[str]) -> None:
    """Simulate streaming: one token (word) per second until cancelled or exhausted."""
    try:
        for w in words:
            await asyncio.sleep(1)
            buf = append_word(w)
            log(
                scene_id,
                f"speaker={speaker_var.get()} stream +{w!r} -> buffer={buf!r}",
            )
    except asyncio.CancelledError:
        log(scene_id, "Actor A: received cancellation, cleaning up and re-raising")
        raise


async def listener_b(
    scene_id: str,
    actor_a_task: asyncio.Task[None],
    handoff_event: asyncio.Event,
    b_words: list[str],
) -> None:
    """
    Poll Actor A's buffer every 2s. On trigger: set Event, cancel A, stream B.
    """
    try:
        while True:
            await asyncio.sleep(2)
            buf = get_buffer()
            log(scene_id, f"[Listener] poll speaker={speaker_var.get()} buffer={buf!r}")

            if TRIGGER in buf.lower():
                log(scene_id, f"[Listener] trigger {TRIGGER!r} detected — firing handoff Event")
                handoff_event.set()
                set_speaker("B")
                actor_a_task.cancel()
                try:
                    await actor_a_task
                except asyncio.CancelledError:
                    # Propagated from awaiting the cancelled task — swallow here so B continues.
                    pass
                log(scene_id, "Actor B: taking over stream after cancel")
                for w in b_words:
                    await asyncio.sleep(1)
                    nbuf = append_word(w)
                    log(
                        scene_id,
                        f"speaker={speaker_var.get()} stream +{w!r} -> buffer={nbuf!r}",
                    )
                return

            if actor_a_task.done():
                err = actor_a_task.exception()
                if err is not None:
                    log(scene_id, f"[Listener] Actor A ended with error: {err!r}")
                    raise err
                log(scene_id, "[Listener] Actor A finished without trigger; stopping listener")
                return
    except asyncio.CancelledError:
        log(scene_id, "Listener B: cancelled")
        raise


async def run_scene(scene_id: str, a_script: list[str], b_script: list[str]) -> None:
    """
    One tenant/session: reset ContextVars for this context, run A + B.

    Await order: we await the listener task first. It either hands off (cancel + B stream)
    or exits if A ends without trigger. Then we drain Actor A if it is still running
    (should only happen if A ended naturally without handoff).
    """
    buffer_var.set(BufferState())
    speaker_var.set("A")
    handoff_event = asyncio.Event()

    task_a = asyncio.create_task(actor_a(scene_id, a_script), name=f"{scene_id}-A")
    task_b = asyncio.create_task(
        listener_b(scene_id, task_a, handoff_event, b_script),
        name=f"{scene_id}-B",
    )

    await task_b

    if not task_a.done():
        task_a.cancel()
        try:
            await task_a
        except asyncio.CancelledError:
            pass

    # Coordinator observes the handoff signal (explicit use of the Event).
    if handoff_event.is_set():
        log(scene_id, "Scene complete: handoff Event was set (B interrupted A)")
    else:
        log(scene_id, "Scene complete: no handoff (A finished without trigger)")


async def main() -> None:
    # Different scripts so "apple" lands at different times — proves concurrent isolation.
    scene1_a = ["hello", "from", "scene", "one", "apple", "would", "be", "cut"]
    scene1_b = ["B", "interrupts", "scene", "one", "done"]
    scene2_a = ["alpha", "beta", "gamma", "delta", "epsilon", "apple", "zeta"]
    scene2_b = ["scene", "two", "takeover", "ok"]

    ctx1 = contextvars.copy_context()
    ctx2 = contextvars.copy_context()

    t1 = ctx1.run(lambda: asyncio.create_task(run_scene("scene-1", scene1_a, scene1_b)))
    t2 = ctx2.run(lambda: asyncio.create_task(run_scene("scene-2", scene2_a, scene2_b)))

    await asyncio.gather(t1, t2)
    log("main", "Both scenes finished — check logs: buffers never crossed scenes")


if __name__ == "__main__":
    if sys.version_info < (3, 11):
        print("Python 3.11+ recommended (asyncio task names / typing)", file=sys.stderr)
    asyncio.run(main())
