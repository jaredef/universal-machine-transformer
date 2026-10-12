#!/usr/bin/env python3
"""A Turing machine that runs any elementary cellular automaton, over the symbols 0 1 _ on a one-way tape.

The tape starts as `_1`: a sentinel at cell 0, then the first row. Each generation is one left-to-right sweep. The
machine carries the old values of the two cells behind it (states cAB) and, reading the old value X of the current
cell, writes the new value of the cell BEHIND it, f(A, B, X), into the current cell. So every row is written one cell to
the right of where it was computed; since a row also widens by one cell on each side per generation, every row starts
at cell 1. After the last cell, two more steps write the row's new right edge (states eB), then the head walks back to
the sentinel (state back) and enters state g, which marks a finished generation, then starts the next sweep.

8 states, 18 rules, for any of the 256 rules.
"""

def rule_table(rule: int):
    """f(a, b, c) for an elementary cellular automaton, Wolfram numbering."""
    return {(a, b, c): (rule >> (a * 4 + b * 2 + c)) & 1 for a in (0, 1) for b in (0, 1) for c in (0, 1)}


def ca_rules(rule: int) -> list[str]:
    f = rule_table(rule)
    R = ["g _ -> c00 _ R"]
    for a in (0, 1):
        for b in (0, 1):
            for x in (0, 1):
                R.append(f"c{a}{b} {x} -> c{b}{x} {f[(a, b, x)]} R")
            R.append(f"c{a}{b} _ -> e{b} {f[(a, b, 0)]} R")              # the first cell past the old row
    for b in (0, 1):
        R.append(f"e{b} _ -> back {f[(b, 0, 0)]} L")                       # the second
    R += ["back 0 -> back 0 L", "back 1 -> back 1 L", "back _ -> g _ S"]
    return R


def ca_source(rule: int, tape: str = "_1") -> str:
    return "blank _\nstart g\naccept qa\nreject qr\ntape " + tape + "\n---\n" + "\n".join(ca_rules(rule)) + "\n"


def reference_rows(rule: int, generations: int) -> list[str]:
    """Row g has width 2g+1: the light cone of a single 1 on a background of 0."""
    f = rule_table(rule)
    row = "1"
    rows = [row]
    for _ in range(generations - 1):
        padded = "00" + row + "00"
        row = "".join(str(f[(int(padded[i - 1]), int(padded[i]), int(padded[i + 1]))]) for i in range(1, len(padded) - 1))
        rows.append(row)
    return rows


def tm_rows(rule: int, generations: int) -> tuple[list[str], int]:
    """Run the Turing machine directly (one-way tape); snapshot the row each time it enters g. Returns rows, steps."""
    rules = {}
    for line in ca_rules(rule):
        s, r, _, n, w, m = line.split()
        rules[(s, r)] = (n, w, m)
    tape, head, state, steps, rows = {0: "_", 1: "1"}, 0, "g", 0, []
    while True:
        if state == "g":
            cells = [tape.get(i, "_") for i in range(1, max(tape) + 1)]
            rows.append("".join(cells).rstrip("_"))
            if len(rows) == generations:
                return rows, steps
        state, w, m = rules[(state, tape.get(head, "_"))]
        tape[head] = w
        head = max(0, head + {"L": -1, "R": 1, "S": 0}[m])
        steps += 1


if __name__ == "__main__":
    for rule in (90, 30, 110):
        got, steps = tm_rows(rule, 16)
        want = reference_rows(rule, 16)
        print(f"rule {rule}: {len(ca_rules(rule))} rules, 16 generations in {steps} steps, "
              f"{'MATCHES' if got == want else 'DIFFERS'} the direct automaton")
        for r in got[:6]:
            print("   ", r.replace("0", ".").replace("1", "#").center(31))
