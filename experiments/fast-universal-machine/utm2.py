#!/usr/bin/env python3
"""A faster universal Turing machine (version 2), and the encoder for the machines it runs.

Version 1 (machines/lag3/derived/utm.py) writes the next state in unary, as k x's, and moves the current-state mark '*'
to entry k by walking from the rule to the far left end and back once per x. Profiled on the lagging-window decoder,
that is 69% of all the work for an 8-state machine and 89% for a 14-state one, and it grows roughly as the fifth power of
the number of states.

Version 2 finds entry k in binary. Every entry's separator carries a tag, the number of trailing zeros of the entry's
index (entry 0 carries the top level), and the next state is written as the levels of k's 1 bits, highest first. Then
"go forward 2^l entries" is a single walk to the next separator whose tag is at least l: starting at entry 0 and taking
k's 1 bits from the top, every jump starts at a multiple of 2^(l+1) and so lands exactly 2^l entries on. The machine also
remembers whether the mark is left or right of the rule it is reading, so it never walks back to the start of the program
between jumps. The rest of the cycle is version 1's.

The tape:

    *k E0 |d E1 |e E2 |d E3 |f E4 ...  $ c0 c1 ...

    entry   three blocks, for reading 0, 1, _:   W D NEXT ;   (W in a b c, D in L R S)
    NEXT    the levels of the target's 1 bits, highest first (d = level 0 ... k = level 7); empty for state 0;
            A to accept; H to reject after writing.     H ;  alone: no rule (reject without writing)
    tags    after each separator: the level symbol for the entry index's trailing zeros (entry 0: k)

With levels up to 7, a described machine has at most 256 states. The system stays universal: a small universal Turing
machine over 0 1 _ fits in far fewer states, and it can run anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "machines" / "lag0" / "derived"))
import lag0  # noqa: E402

BMAX = 8
LEV = "defghijk"                            # level 0 .. 7
LEVM = "DEFGJKMN"                           # the same levels, marked as used
SIM = ["0", "1", "_"]
MARK = {"0": "O", "1": "I", "_": "U"}
WLET = {"0": "a", "1": "b", "_": "c"}
LETW = {v: k for k, v in WLET.items()}
LOW = {"L": "l", "R": "r", "S": "s"}
ALL = ["|", "*", ";", "a", "b", "c", "L", "R", "S", "l", "r", "s", "A", "H", "$",
       "0", "1", "_", "O", "I", "U"] + list(LEV) + list(LEVM)


def utm_rules() -> list[tuple[str, str, str, str, str]]:
    R: list[tuple[str, str, str, str, str]] = []

    def scan(state, move, stop):
        for s in ALL:
            if s not in stop:
                R.append((state, s, state, s, move))

    # 1 find the described machine's head cell and read it
    scan("find", "R", ("O", "I", "U"))
    for i, s in enumerate(SIM):
        R.append(("find", MARK[s], f"back{i}", MARK[s], "L"))
    # 2 back to the current entry; skip its tag, then the blocks before the one for the symbol read
    for i in range(3):
        scan(f"back{i}", "L", ("*",))
        R.append((f"back{i}", "*", f"at{i}", "*", "R"))
    for k in (1, 2):
        scan(f"at{k}", "R", (";",))
        R.append((f"at{k}", ";", f"at{k - 1}", ";", "R"))
    for lv in LEV:
        R.append(("at0", lv, "at0", lv, "R"))                       # the entry's tag (only when reading 0)
    # 3 read W and D, mark the block, carry them to the head, write and move
    for w in "abc":
        R.append(("at0", w, f"rd_{w}", w, "R"))
        for d in "LRS":
            R.append((f"rd_{w}", d, f"go_{w}{d}", LOW[d], "R"))
            scan(f"go_{w}{d}", "R", ("O", "I", "U"))
            for s in SIM:
                sym = LETW[w]
                if d == "S":
                    R.append((f"go_{w}{d}", MARK[s], "ret", MARK[sym], "L"))
                else:
                    R.append((f"go_{w}{d}", MARK[s], "markR" if d == "R" else "markL", sym, d))
    for s in SIM:
        R.append(("markR", s, "ret", MARK[s], "L"))
        R.append(("markL", s, "ret", MARK[s], "L"))
    R.append(("markL", "$", "markR", "$", "R"))                       # left of cell 0: stay on cell 0
    # 4 back to the marked block, and read NEXT
    scan("ret", "L", ("l", "r", "s"))
    for d in "lrs":
        R.append(("ret", d, "nx", d, "R"))
    R.append(("nx", "A", "qa", "A", "S"))
    for lv in LEV:
        R.append(("nx", lv, "clr", lv, "L"))
    R.append(("nx", ";", "clr", ";", "L"))
    # 5 move '*' to the target: clear it, put it on entry 0, then for each level of NEXT jump right to the next
    #   separator tagged at least that level. f says where the mark is: L(eft) or R(ight) of the rule being read.
    scan("clr", "L", ("*",))
    R.append(("clr", "*", "toL", "|", "L"))
    scan("toL", "L", ("_",))
    R.append(("toL", "_", "set0", "_", "R"))
    R.append(("set0", "|", "toruleL", "*", "R"))
    scan("toruleL", "R", ("l", "r", "s"))
    scan("toruleR", "L", ("l", "r", "s"))
    for f in "LR":
        for d in "lrs":
            R.append((f"torule{f}", d, f"rn{f}", d, "R"))
        for m in LEVM:
            R.append((f"rn{f}", m, f"rn{f}", m, "R"))
        for i, lv in enumerate(LEV):
            R.append((f"rn{f}", lv, f"tonav{i}{f}", LEVM[i], "L" if f == "L" else "R"))
        R.append((f"rn{f}", ";", "fin", ";", "L"))
    for i in range(BMAX):
        scan(f"tonav{i}L", "L", ("*",))
        scan(f"tonav{i}R", "R", ("*",))
        for f in "LR":
            R.append((f"tonav{i}{f}", "*", f"jmp{i}{f}", "|", "R"))
            st = f"jmp{i}{f}"
            for s in ALL:
                if s == "|":
                    R.append((st, s, f"jt{i}{f}", s, "R"))
                elif s in ("l", "r", "s"):
                    R.append((st, s, f"jmp{i}R", s, "R"))             # passing the rule: the mark is now right of it
                else:
                    R.append((st, s, st, s, "R"))
            for t, lv in enumerate(LEV):
                R.append((f"jt{i}{f}", lv, f"put{f}" if t >= i else st, lv, "L" if t >= i else "R"))
    for f in "LR":
        R.append((f"put{f}", "|", f"torule{f}", "*", "R" if f == "L" else "L"))
    # 6 restore the used levels and the block's mark, then start the next cycle
    for i in range(BMAX):
        R.append(("fin", LEVM[i], "fin", LEV[i], "L"))
    for d in "LRS":
        R.append(("fin", LOW[d], "find", d, "R"))
    return R


def tz(j: int) -> int:
    return (j & -j).bit_length() - 1


def states_of(prog: lag0.Program) -> list[str]:
    named = {s for (s, _) in prog.rules} | {n for (n, _, _) in prog.rules.values()}
    return [prog.start] + sorted(named - {prog.start, prog.accept, prog.reject})


def encode(prog: lag0.Program, tape: str, order: list[str] | None = None) -> str:
    """The universal machine's input: the description of `prog`, then `tape` with the head on cell 0. `order` may
    place the states differently (the start state must stay first)."""
    states = order or states_of(prog)
    if len(states) > 2 ** BMAX:
        raise ValueError(f"at most {2 ** BMAX} states")
    index = {s: i for i, s in enumerate(states)}
    entries = []
    for j, q in enumerate(states):
        tag = LEV[BMAX - 1] if j == 0 else LEV[tz(j)]
        blocks = []
        for sym in SIM:
            rule = prog.rules.get((q, sym))
            if rule is None:
                blocks.append("H;")
                continue
            nxt, write, move = rule
            if nxt == prog.accept:
                target = "A"
            elif nxt == prog.reject:
                target = "H"
            else:
                k = index[nxt]
                target = "".join(LEV[l] for l in range(BMAX - 1, -1, -1) if k >> l & 1)
            blocks.append(WLET[write] + move + target + ";")
        entries.append(tag + "".join(blocks))
    cells = list(tape) or ["_"]
    cells[0] = MARK[cells[0]]
    return "*" + "|".join(entries) + "$" + "".join(cells)


def utm_source(tape: str) -> str:
    rules = "\n".join(f"{s} {r} -> {n} {w} {m}" for s, r, n, w, m in utm_rules())
    return f"blank _\nstart find\naccept qa\nreject qr\ntape {tape}\n---\n{rules}\n"


if __name__ == "__main__":
    rules = utm_rules()
    keys = [(s, r) for s, r, _, _, _ in rules]
    assert len(keys) == len(set(keys)), "a state reads a symbol twice"
    states = {r[0] for r in rules} | {r[2] for r in rules}
    print(f"universal machine version 2: {len(states)} states, {len(ALL)} symbols, {len(rules)} rules")
