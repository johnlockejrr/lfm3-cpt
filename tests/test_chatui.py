"""Reasoning-aware display and history in `chat`."""

import pytest

from lfm3_cpt.chatui import SEPARATOR, AnswerStreamer, final_answer, visible_text

# What the original LFM3 printed with thinking off: answer, a stray </think>, the answer again.
STRAY = "Ankara'dır.</think>Ankara'dır."


@pytest.mark.parametrize(
    "full, show, final, expected",
    [
        ("plain answer", False, True, "plain answer"),
        ("<think>reasoning</think>answer", False, True, "answer"),
        ("<think>reasoning</think>\nanswer", True, True, "reasoning" + SEPARATOR + "answer"),
        ("<think>still thinking", False, False, ""),
        ("<think>still thinking", True, False, "still thinking"),
        (STRAY, False, True, "Ankara'dır." + SEPARATOR + "Ankara'dır."),
        ("Ankara'dır.</thi", False, False, "Ankara'dır."),  # partial tag held back while streaming
        ("a < b", False, True, "a < b"),
    ],
)
def test_visible_text(full, show, final, expected):
    assert visible_text(full, show, final) == expected


def test_final_answer_keeps_only_the_answer():
    assert final_answer(STRAY) == "Ankara'dır."
    assert final_answer("<think>x</think>  the answer ") == "the answer"
    assert final_answer("no tags ") == "no tags"
    assert final_answer("<think>never finished") == ""


class _Tok:  # TextStreamer only stores the tokenizer; we feed text chunks directly
    pass


def _stream(chunks, show=False):
    out, flags = [], []
    s = AnswerStreamer(_Tok(), show_thinking=show, write=out.append, on_thinking=lambda: flags.append(1))
    for i, c in enumerate(chunks):
        s.on_finalized_text(c, stream_end=i == len(chunks) - 1)
    return "".join(out), flags


def test_streamer_hides_reasoning_and_handles_split_tags():
    text, flags = _stream(["<thi", "nk>let me ", "think</th", "ink>The answer", " is 19."])
    assert text == "The answer is 19.\n" and flags == [1]  # "(thinking…)" announced once
    text, _ = _stream(["Ankara'dır.</", "think>Ankara'dır."])
    assert text == "Ankara'dır." + SEPARATOR + "Ankara'dır.\n"
    text, flags = _stream(["Just an ", "answer."])
    assert text == "Just an answer.\n" and flags == []
    text, _ = _stream(["<think>a", "b</think>c"], show=True)
    assert text == "ab" + SEPARATOR + "c\n"
