"""Display and history handling for reasoning models in `chat`.

LFM3 can reason before answering, between `<think>` and `</think>`. Even with thinking off, the original model
sometimes writes its answer, then `</think>`, then the answer again. This module decides what is shown while
streaming and what is kept as the assistant's turn in the conversation history:

- output that starts with `<think>`: the reasoning is hidden (or shown with --thinking), the answer is streamed;
- a stray `</think>` in the middle: shown as a visible separator, and only the text after it is kept;
- partial tags at the end of a chunk are held back, so `</thi` never flashes on screen.
"""

from __future__ import annotations

from collections.abc import Callable

from transformers import TextStreamer

OPEN, CLOSE = "<think>", "</think>"
SEPARATOR = "\n— answer —\n"


def _held_back(full: str) -> int:
    """Length of the longest suffix of `full` that could be the start of a tag."""
    for n in range(min(len(CLOSE), len(full)), 0, -1):
        tail = full[-n:]
        if CLOSE.startswith(tail) or OPEN.startswith(tail):
            return n
    return 0


def visible_text(full: str, show_thinking: bool = False, final: bool = False) -> str:
    """What the user should see of the generated text so far."""
    if not final:
        full = full[: len(full) - _held_back(full)]
    stripped = full.lstrip()
    if stripped.startswith(OPEN):
        body = stripped[len(OPEN) :]
        if CLOSE not in body:
            return body if show_thinking else ""
        thinking, answer = body.split(CLOSE, 1)
        return (thinking + SEPARATOR if show_thinking else "") + answer.lstrip("\n")
    return full.replace(CLOSE, SEPARATOR)


def final_answer(full: str) -> str:
    """The assistant turn to keep in the history: the text after the last `</think>`."""
    if CLOSE in full:
        return full.rsplit(CLOSE, 1)[1].strip()
    stripped = full.strip()
    return "" if stripped.startswith(OPEN) else stripped  # unterminated reasoning: no answer


class AnswerStreamer(TextStreamer):
    """A TextStreamer that applies `visible_text` incrementally."""

    def __init__(
        self,
        tokenizer,
        show_thinking: bool = False,
        write: Callable[[str], None] | None = None,
        on_thinking: Callable[[], None] | None = None,
    ):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.show_thinking = show_thinking
        self.write = write or (lambda s: print(s, end="", flush=True))
        self.on_thinking = on_thinking
        self.reset()

    def reset(self) -> None:
        self.full, self.printed, self.thinking_announced = "", "", False

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        self.full += text
        if (
            not self.show_thinking
            and not self.thinking_announced
            and self.full.lstrip().startswith(OPEN)
            and self.on_thinking
        ):
            self.on_thinking()
            self.thinking_announced = True
        visible = visible_text(self.full, self.show_thinking, final=stream_end)
        if visible.startswith(self.printed):
            new = visible[len(self.printed) :]
            if new:
                self.write(new)
                self.printed = visible
        if stream_end:
            self.write("\n")
