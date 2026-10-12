#!/usr/bin/env python3
"""lag3 core: the universal machine (utm.py) on the lagging-window decoder.

The step function of the decoder can be the universal machine's exact rule (lag0.rule), lag1's derived weights, or a
trained network. Everything here is shared by the training script (fixtures/train_lag3.ref.py) and the witness.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parents[1] / "lag0" / "derived", HERE.parents[1] / "lag1" / "derived"):
    sys.path.insert(0, str(p))
import lag0  # noqa: E402
import utm  # noqa: E402

B = lag0.BOUNDARY

# Machines the network is tested on. None of them is in its training data, which is random machines only.
HELD_OUT = {
    "increment": ("add one to a binary number (with a leading 0)", """
q0 0 -> q0 0 R
q0 1 -> q0 1 R
q0 _ -> q1 _ L
q1 1 -> q1 0 L
q1 0 -> qa 1 S
q1 _ -> qa 1 S"""),
    "decrement": ("subtract one from a binary number", """
q0 0 -> q0 0 R
q0 1 -> q0 1 R
q0 _ -> q1 _ L
q1 0 -> q1 1 L
q1 1 -> qa 0 S"""),
    "invert": ("flip every bit", """
q0 0 -> q0 1 R
q0 1 -> q0 0 R
q0 _ -> qa _ S"""),
    "parity": ("accept if the number of 1s is even", """
e 0 -> e 0 R
e 1 -> o 1 R
e _ -> qa _ S
o 0 -> o 0 R
o 1 -> e 1 R
o _ -> qr _ S"""),
    "div3": ("accept if the binary number is a multiple of 3", """
r0 0 -> r0 0 R
r0 1 -> r1 1 R
r0 _ -> qa _ S
r1 0 -> r2 0 R
r1 1 -> r0 1 R
r1 _ -> qr _ S
r2 0 -> r1 0 R
r2 1 -> r2 1 R
r2 _ -> qr _ S"""),
    "unary-add": ("add two unary numbers: 111_11 becomes 11111", """
q0 1 -> q0 1 R
q0 _ -> q1 1 R
q1 1 -> q1 1 R
q1 _ -> q2 _ L
q2 1 -> qa _ S"""),
    "palindrome": ("accept if the bits read the same backwards (six states)", """
q0 0 -> q1 _ R
q0 1 -> q2 _ R
q0 _ -> qa _ S
q1 0 -> q1 0 R
q1 1 -> q1 1 R
q1 _ -> q3 _ L
q2 0 -> q2 0 R
q2 1 -> q2 1 R
q2 _ -> q4 _ L
q3 0 -> q5 _ L
q3 1 -> qr 1 S
q3 _ -> qa _ S
q4 1 -> q5 _ L
q4 0 -> qr 0 S
q4 _ -> qa _ S
q5 0 -> q5 0 L
q5 1 -> q5 1 L
q5 _ -> q0 _ R"""),
}
# palindrome's start state must be q0 and the program must begin at the left end of the input; its erasing
# leaves blanks behind, which the one-way tape handles because the machine never moves left of cell 0.


def held_out_source(name: str, tape: str) -> str:
    start = "e" if name == "parity" else "r0" if name == "div3" else "q0"
    return f"blank _\nstart {start}\naccept qa\nreject qr\ntape {tape}\n---{HELD_OUT[name][1]}\n"


def random_source(rng: random.Random, n_states: int, tape: str, moves: str = "LRS", halt: float = 1.0) -> str:
    """A random machine over 0 1 _ with n_states states (plus accept and reject); some cases have no rule.
    `moves` is the bag a move is drawn from (e.g. "RRRLS" for machines that tend to travel right); `halt` scales how
    often a rule halts."""
    states = [f"s{i}" for i in range(n_states)]
    lines = []
    for q in states:
        for sym in utm.SIM:
            if rng.random() < 0.12:
                continue
            nxt = rng.choice(states) if rng.random() > 0.25 * halt else rng.choice(["qa", "qa", "qr"])
            lines.append(f"{q} {sym} -> {nxt} {rng.choice(utm.SIM)} {rng.choice(moves)}")
    return f"blank _\nstart s0\naccept qa\nreject qr\ntape {tape}\n---\n" + "\n".join(lines) + "\n"


def reference(src: str, max_steps: int = 100_000) -> tuple[str, str]:
    """Run a machine directly, on a one-way tape (a left move on cell 0 stays). Returns (verdict, tape)."""
    p = lag0.parse_tm0(src)
    tape = {i: s for i, s in enumerate(p.tape) if s != p.blank}
    head, state = 0, p.start
    for _ in range(max_steps):
        if state == p.accept:
            return "ACCEPT", fmt(tape, p.blank)
        if state == p.reject:
            return "REJECT", fmt(tape, p.blank)
        rule = p.rules.get((state, tape.get(head, p.blank)))
        if rule is None:
            return "REJECT", fmt(tape, p.blank)
        state, write, move = rule
        if write == p.blank:
            tape.pop(head, None)
        else:
            tape[head] = write
        head = max(0, head + {"L": -1, "R": 1, "S": 0}[move])
    return "TIMEOUT", fmt(tape, p.blank)


def fmt(tape: dict, blank: str) -> str:
    if not tape:
        return blank
    return "".join(tape.get(i, blank) for i in range(0, max(tape) + 1))


def utm_program(src: str) -> lag0.Program:
    """The universal machine, with the described machine `src` and its input encoded on the tape."""
    p = lag0.parse_tm0(src)
    return lag0.parse_tm0(utm.utm_source(utm.encode(p, "".join(p.tape))))


class Exact:
    """The universal machine's exact rule, memoized per window."""

    def __init__(self, prog: lag0.Program):
        self.prog, self.cache = prog, {}

    def __call__(self, w):
        if w not in self.cache:
            self.cache[w] = lag0.rule(self.prog, *w)
        return self.cache[w]


def lag_run(prog: lag0.Program, step, max_decode: int = 20_000_000, seen: set | None = None):
    """Run the lagging-window decoder with `step(window) -> tokens or None`. Returns (verdict, M's tape, decode steps)."""
    ring = lag0.initial_ring(prog)
    T = [ring[-1], *ring]
    k = left_grown = 0
    for n in range(max_decode):
        w = (T[k], T[k + 1], T[k + 2])
        if seen is not None:
            seen.add(w)
        out = step(w)
        if not out:
            h = lag0.split_head(w[1])
            verdict = "ACCEPT" if h and h[0] == prog.accept else "REJECT"
            final = lag0.read_tape(prog, T[k + 1:], left_grown)
            return verdict, utm.decode_tape(final), n
        if w[1] == B and len(out) == 2:
            left_grown += 1
        T.extend(out)
        k += 1
    return "TIMEOUT", "", max_decode
