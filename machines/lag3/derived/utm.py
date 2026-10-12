#!/usr/bin/env python3
"""A universal Turing machine, written as a .tm0 program, and the encoder for the machines it runs.

The tape holds a described machine M, then M's tape:

    *E0 |E1 |E2 ... $ c0 c1 c2 ...

    Ei      the entry for M's state i: three blocks, for reading 0, 1, and _ in that order
    block   W D NEXT ;      W in a b c  (write 0, 1, _),  D in L R S,
                            NEXT = x^k (go to state k), A (accept), or H (reject after writing)
            H ;             no rule for this case: reject without writing
    *       marks the entry of M's current state (the others start with |)
    $       separates the description from M's tape
    c       M's tape cells, 0 1 _ ; the cell under M's head is marked O I U

One cycle of the universal machine is one step of M: find the marked cell and read it; go to the current entry and
select the block for that symbol; read W and D and mark the block; go back, write W and move the mark; return to the
block and read NEXT; move '*' to entry k by counting the x's. M's tape is one-way infinite to the right: a left move
on cell 0 stays put.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lag0" / "derived"))
import lag0  # noqa: E402

SIM = ["0", "1", "_"]                       # M's symbols, in block order
MARK = {"0": "O", "1": "I", "_": "U"}
WLET = {"0": "a", "1": "b", "_": "c"}
LETW = {v: k for k, v in WLET.items()}
LOW = {"L": "l", "R": "r", "S": "s"}
ALL = ["|", "*", ";", "a", "b", "c", "L", "R", "S", "l", "r", "s", "x", "y", "A", "H", "$",
       "0", "1", "_", "O", "I", "U"]


def utm_rules() -> list[tuple[str, str, str, str, str]]:
    R: list[tuple[str, str, str, str, str]] = []

    def scan(state, move, stop, also=()):
        """`state` moves `move` over every symbol except those in `stop` (handled by the caller)."""
        for s in ALL:
            if s not in stop:
                R.append((state, s, state, s, move))

    # find M's head cell and read it
    scan("find", "R", ("O", "I", "U"))
    for i, s in enumerate(SIM):
        R.append(("find", MARK[s], f"back{i}", MARK[s], "L"))
    # back to the current entry, then skip to the block for the symbol read
    for i in range(3):
        scan(f"back{i}", "L", ("*",))
        R.append((f"back{i}", "*", f"at{i}", "*", "R"))
    for k in (1, 2):
        scan(f"at{k}", "R", (";",))
        R.append((f"at{k}", ";", f"at{k - 1}", ";", "R"))
    # at the block: read W (H here = no rule: no transition, so the machine rejects)
    for w in "abc":
        R.append(("at0", w, f"rd_{w}", w, "R"))
        for d in "LRS":
            R.append((f"rd_{w}", d, f"go_{w}{d}", LOW[d], "R"))      # mark the selected block by lowering D
            scan(f"go_{w}{d}", "R", ("O", "I", "U"))
            for s in SIM:
                sym = LETW[w]
                if d == "S":
                    R.append((f"go_{w}{d}", MARK[s], "ret", MARK[sym], "L"))
                else:
                    R.append((f"go_{w}{d}", MARK[s], "markR" if d == "R" else "markL", sym, d))
    # move M's head mark
    for s in SIM:
        R.append(("markR", s, "ret", MARK[s], "L"))
        R.append(("markL", s, "ret", MARK[s], "L"))
    R.append(("markL", "$", "markR", "$", "R"))                       # left of cell 0: stay on cell 0
    # back to the marked block, and read NEXT
    scan("ret", "L", ("l", "r", "s"))
    for d in "lrs":
        R.append(("ret", d, "nx", d, "R"))
    R.append(("nx", "A", "qa", "A", "S"))                             # accept
    R.append(("nx", "x", "clr", "x", "L"))
    R.append(("nx", ";", "clr", ";", "L"))
    # move '*' to entry k: clear it, put it on entry 0, then advance it once per x
    scan("clr", "L", ("*",))
    R.append(("clr", "*", "toL", "|", "L"))
    scan("toL", "L", ("_",))
    R.append(("toL", "_", "set0", "_", "R"))
    R.append(("set0", "|", "cnt", "*", "R"))
    scan("cnt", "R", ("l", "r", "s"))
    for d in "lrs":
        R.append(("cnt", d, "cx", d, "R"))
    R.append(("cx", "y", "cx", "y", "R"))
    R.append(("cx", "x", "adv", "y", "L"))
    R.append(("cx", ";", "fin", ";", "L"))
    scan("adv", "L", ("_",))
    R.append(("adv", "_", "adv2", "_", "R"))
    scan("adv2", "R", ("*",))
    R.append(("adv2", "*", "adv3", "|", "R"))
    scan("adv3", "R", ("|",))
    R.append(("adv3", "|", "tol2", "*", "L"))
    scan("tol2", "L", ("_",))
    R.append(("tol2", "_", "cnt", "_", "R"))
    # restore the counted x's and the block's mark, then start the next cycle
    R.append(("fin", "y", "fin", "x", "L"))
    for d in "LRS":
        R.append(("fin", LOW[d], "find", d, "R"))
    return R


def encode(prog: lag0.Program, tape: str) -> str:
    """The universal machine's input: the description of `prog`, then `tape` with the head on cell 0."""
    named = {s for (s, _) in prog.rules} | {n for (n, _, _) in prog.rules.values()}
    states = [prog.start] + sorted(named - {prog.start, prog.accept, prog.reject})   # every state gets an entry
    index = {s: i for i, s in enumerate(states)}
    entries = []
    for q in states:
        blocks = []
        for sym in SIM:
            rule = prog.rules.get((q, sym))
            if rule is None:
                blocks.append("H;")
                continue
            nxt, write, move = rule
            if write not in SIM:
                raise ValueError(f"symbol {write!r} is not one of 0 1 _")
            target = "A" if nxt == prog.accept else "H" if nxt == prog.reject else "x" * index[nxt]
            blocks.append(WLET[write] + move + target + ";")
        entries.append("".join(blocks))
    cells = list(tape) or ["_"]
    cells[0] = MARK[cells[0]]
    return "*" + "|".join(entries) + "$" + "".join(cells)


def utm_source(tape: str) -> str:
    rules = "\n".join(f"{s} {r} -> {n} {w} {m}" for s, r, n, w, m in utm_rules())
    return f"blank _\nstart find\naccept qa\nreject qr\ntape {tape}\n---\n{rules}\n"


def decode_tape(final: str) -> str:
    """M's tape, from the universal machine's final tape, in tm0's format."""
    cells = final.split("$", 1)[1] if "$" in final else ""
    cells = "".join({v: k for k, v in MARK.items()}.get(c, c) for c in cells).rstrip("_")
    return cells or "_"


if __name__ == "__main__":
    rules = utm_rules()
    states = {r[0] for r in rules} | {r[2] for r in rules}
    print(f"universal machine: {len(states)} states, {len(ALL)} symbols, {len(rules)} rules")
