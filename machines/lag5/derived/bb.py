#!/usr/bin/env python3
"""The busy beaver champions, as machines the universal machine can run.

A busy beaver is the n-state, two-symbol Turing machine that, started on a blank tape, runs longest before halting.
The champions here are the proven ones: BB(2), 6 steps leaving 4 ones, and BB(4), 107 steps leaving 13 ones (Brady,
1983). They use a two-way tape; the universal machine's described tape is one-way (a left move on cell 0 stays). So a
one-state prelude walks right over `pad` cells, erasing them, and hands over to the champion's start state, which then
has `pad` cells of room on its left: exactly as far left as the champion ever goes.
"""

from __future__ import annotations

# state -> ((write, move, next) on blank, (write, move, next) on 1); next "H" halts. Blank is "_" (the usual 0).
CHAMPIONS = {
    "bb2": {"A": (("1", "R", "B"), ("1", "L", "B")),
            "B": (("1", "L", "A"), ("1", "R", "H"))},
    "bb4": {"A": (("1", "R", "B"), ("1", "L", "B")),
            "B": (("1", "L", "A"), ("_", "L", "C")),
            "C": (("1", "R", "H"), ("1", "L", "D")),
            "D": (("1", "R", "D"), ("_", "R", "A"))},
}
EXPECTED = {"bb2": (6, 4), "bb4": (107, 13)}               # (steps, ones), the proven values


def run_two_way(name: str):
    """Run a champion directly on a two-way blank tape. Returns (steps, ones, leftmost, rightmost, history), where the
    history holds (state, head, tape) before each step and after the last, positions relative to the start cell."""
    m, tape, head, state, steps = CHAMPIONS[name], {}, 0, "A", 0
    lo = hi = 0
    history = [(state, head, dict(tape))]
    while state != "H":
        write, move, state = m[state][tape.get(head, "_") == "1"]
        if write == "1":
            tape[head] = "1"
        else:
            tape.pop(head, None)
        head += -1 if move == "L" else 1
        lo, hi, steps = min(lo, head), max(hi, head), steps + 1
        history.append((state, head, dict(tape)))
    return steps, len(tape), lo, hi, history


def pad(name: str) -> int:
    return -run_two_way(name)[2]


def bb_rules(name: str) -> list[str]:
    """The prelude (state w) and the champion, in the universal machine's rule syntax; halting accepts."""
    R = ["w 0 -> w _ R", "w _ -> A _ S"]
    for q, cases in CHAMPIONS[name].items():
        for read, (write, move, nxt) in zip("_1", cases):
            R.append(f"{q} {read} -> {'qa' if nxt == 'H' else nxt} {write} {move}")
    return R


def bb_source(name: str) -> str:
    return ("blank _\nstart w\naccept qa\nreject qr\ntape " + "0" * pad(name) + "\n---\n"
            + "\n".join(bb_rules(name)) + "\n")


if __name__ == "__main__":
    for name in CHAMPIONS:
        steps, ones, lo, hi, _ = run_two_way(name)
        print(f"{name}: steps={steps} ones={ones} cells {lo}..{hi} pad={pad(name)} "
              f"{'OK' if (steps, ones) == EXPECTED[name] else 'MISMATCH'}")
        print(bb_source(name))
