#!/usr/bin/env python3
"""lag0: a Turing machine run by a lagging-window decoder (a Lag system).

The machine is an autoregressive loop with no hidden state:

    transcript   unbounded, append-only            (the tape)
    window       three tokens, read at a moving index (the head's view)
    rule table   a finite map  window -> tokens    (the weights)
    a step       read the window, append the output, advance the window by one

`compile_rules` turns a `.tm0` program into that finite rule table. A configuration
of the Turing machine is written into the transcript as a ring of cells,

    #  c1  c2  ...  cn          one cell holds the head:  state:symbol

and each step rewrites the middle cell of the window from its two neighbours, so one
full pass of the window around the ring is exactly one Turing step. The window lags
behind the frontier: it reads the old end of the transcript while the new
configuration is appended at the far end. Nothing is ever deleted; every
configuration the machine passes through stays in the transcript.

The negative witness is `--discipline recency`: the same rule table, but the window
holds the most recent three tokens, as in ordinary decoding. The transcript is then
never read again, the loop is finite-state, and it cannot compute.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

BOUNDARY = "#"
SEP = ":"

# ---------------------------------------------------------------- the .tm0 program


@dataclass
class Program:
    blank: str
    start: str
    accept: str
    reject: str
    tape: list[str]
    rules: dict[tuple[str, str], tuple[str, str, str]]  # (state, read) -> (next, write, move)


def parse_tm0(text: str) -> Program:
    header: dict[str, str] = {}
    rules: dict[tuple[str, str], tuple[str, str, str]] = {}
    in_rules = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "---":
            in_rules = True
            continue
        parts = line.split()
        if not in_rules:
            header[parts[0]] = " ".join(parts[1:])
            continue
        state, read, arrow, nxt, write, move = parts
        if arrow != "->" or move not in ("L", "R", "S"):
            raise ValueError(f"malformed rule: {line!r}")
        rules[(state, read)] = (nxt, write, move)
    return Program(header["blank"], header["start"], header["accept"], header["reject"], list(header["tape"]), rules)


# ---------------------------------------------------------------- tokens


def head(state: str, symbol: str) -> str:
    return f"{state}{SEP}{symbol}"


def split_head(token: str) -> tuple[str, str] | None:
    if SEP in token and token != BOUNDARY:
        state, symbol = token.split(SEP, 1)
        return state, symbol
    return None


# ---------------------------------------------------------------- compiling the rule table


def step_of(prog: Program, token: str) -> tuple[str, str, str] | None:
    """The Turing transition a head token will take, or None if it is halted or stuck."""
    h = split_head(token)
    if h is None:
        return None
    state, symbol = h
    if state in (prog.accept, prog.reject):
        return None
    return prog.rules.get((state, symbol))


def rule(prog: Program, left: str, mid: str, right: str) -> list[str] | None:
    """New value of the middle cell, given its two neighbours. None means: no rule (halt)."""
    blank = prog.blank
    if mid == BOUNDARY:
        t = step_of(prog, right)
        if t and t[2] == "L":                       # the head walks off the left end: grow the tape
            return [BOUNDARY, head(t[0], blank)]
        return [BOUNDARY]
    h = split_head(mid)
    if h is not None:
        t = step_of(prog, mid)
        if t is None:                               # accept, reject, or a missing transition
            return None
        nxt, write, move = t
        if move == "S":
            return [head(nxt, write)]
        if move == "R" and right == BOUNDARY:       # the head walks off the right end: grow the tape
            return [write, head(nxt, blank)]
        return [write]
    tl, tr = step_of(prog, left), step_of(prog, right)
    if tl and tl[2] == "R":                          # the head arrives from the left
        return [head(tl[0], mid)]
    if tr and tr[2] == "L":                          # the head arrives from the right
        return [head(tr[0], mid)]
    return [mid]


def alphabet(prog: Program) -> list[str]:
    symbols = {prog.blank, *prog.tape}
    states = {prog.start, prog.accept, prog.reject}
    for (s, r), (n, w, _) in prog.rules.items():
        symbols |= {r, w}
        states |= {s, n}
    symbols, states = sorted(symbols), sorted(states)
    return [BOUNDARY, *symbols, *(head(q, a) for q in states for a in symbols)]


def compile_rules(prog: Program) -> dict[tuple[str, str, str], tuple[str, ...]]:
    """The finite rule table. Windows are triples of tokens; a ring of one cell lets the left and right of a window
    be the same head, so only the middle and its neighbours are kept distinct."""
    table: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for window in product(alphabet(prog), repeat=3):
        heads = [split_head(t) is not None for t in window]
        if heads[1] and (heads[0] or heads[2]):     # a cell beside the head is never a second head
            continue
        out = rule(prog, *window)
        if out is not None:
            table[window] = tuple(out)
    return table


# ---------------------------------------------------------------- running


@dataclass
class Result:
    verdict: str
    tm_steps: int
    decode_steps: int
    transcript: list[str]
    window_at: int
    tape: str


def initial_ring(prog: Program) -> list[str]:
    cells = list(prog.tape) or [prog.blank]
    cells[0] = head(prog.start, cells[0])
    return [BOUNDARY, *cells]


def read_tape(prog: Program, ring: list[str], left_grown: int) -> str:
    """Format the tape as tm0 does: from min(first nonblank, 0) to max(last nonblank, 0)."""
    i = ring.index(BOUNDARY)
    ring = ring[i:] + ring[:i]                      # rotate the ring to start at '#'
    cells = [t for t in ring if t != BOUNDARY]
    cells = [split_head(t)[1] if split_head(t) else t for t in cells]
    marks = {i - left_grown: s for i, s in enumerate(cells) if s != prog.blank}
    if not marks:
        return prog.blank
    lo, hi = min(min(marks), 0), max(max(marks), 0)
    return "".join(marks.get(i, prog.blank) for i in range(lo, hi + 1))


def run(prog: Program, table, discipline: str = "lag", max_steps: int = 100_000, trace=None) -> Result:
    ring = initial_ring(prog)
    transcript = [ring[-1], *ring]                  # the ring, preceded by its last cell
    k = 0                                           # where the window starts
    tm_steps = left_grown = 0
    seen: set[tuple[str, ...]] = set()
    for step in range(max_steps):
        window = tuple(transcript[k:k + 3]) if discipline == "lag" else tuple(transcript[-3:])
        out = table.get(window)
        if trace:
            trace(step, k if discipline == "lag" else len(transcript) - 3, window, out, transcript)
        if out is None:
            if discipline == "recency":             # the loop stalled on a window it has no rule for
                return Result("NO-RULE", tm_steps, step, transcript, len(transcript) - 3, "")
            ring_now = transcript[k + 1:]           # one whole configuration, as a ring
            h = split_head(window[1])
            verdict = "ACCEPT" if h and h[0] == prog.accept else "REJECT"
            return Result(verdict, tm_steps, step, transcript, k, read_tape(prog, ring_now, left_grown))
        if discipline == "recency":
            if window in seen:
                return Result("CYCLE", tm_steps, step, transcript, len(transcript) - 3, "")
            seen.add(window)
        if step_of(prog, window[1]):
            tm_steps += 1
        if window[1] == BOUNDARY and len(out) == 2:
            left_grown += 1
        transcript.extend(out)
        k += 1
    return Result("TIMEOUT", tm_steps, max_steps, transcript, k, "")


# ---------------------------------------------------------------- command line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("program", type=Path)
    ap.add_argument("--discipline", choices=["lag", "recency"], default="lag")
    ap.add_argument("--trace", action="store_true", help="print every decode step")
    ap.add_argument("--passes", action="store_true", help="print each configuration as it is completed")
    ap.add_argument("--rules", action="store_true", help="print the compiled rule table size and exit")
    args = ap.parse_args()

    prog = parse_tm0(args.program.read_text())
    table = compile_rules(prog)
    if args.rules:
        print(f"alphabet={len(alphabet(prog))} rules={len(table)}")
        return 0

    def trace(step, at, window, out, transcript):
        shown = " ".join(window)
        print(f"step={step:<4} window@{at:<4} [{shown}] -> {' '.join(out) if out else '(no rule)'}")

    result = run(prog, table, args.discipline, trace=trace if args.trace else None)
    if args.passes and args.discipline == "lag":
        for n, ring in enumerate(configurations(prog, result.transcript)):
            print(f"pass={n:<3} {' '.join(ring)}")
    if result.verdict in ("CYCLE", "NO-RULE"):
        print(f"{result.verdict} decode_steps={result.decode_steps} transcript={len(result.transcript)} "
              f"tm_steps={result.tm_steps}")
        return 1
    print(f"{result.verdict} steps={result.tm_steps} tape={result.tape} "
          f"decode_steps={result.decode_steps} transcript={len(result.transcript)} rules={len(table)}")
    return 0 if result.verdict == "ACCEPT" else 1


def configurations(prog: Program, transcript: list[str]) -> list[list[str]]:
    """Every complete configuration in the transcript, in order. Each one is written as a ring starting with '#'."""
    ring = initial_ring(prog)
    emitted = transcript[len(ring) + 1:]
    rings, cur = [ring], None
    for t in emitted:
        if t == BOUNDARY:
            if cur is not None:
                rings.append(cur)
            cur = [t]
        elif cur is not None:
            cur.append(t)
    return rings                                    # the last, partial ring is left out


if __name__ == "__main__":
    sys.exit(main())
