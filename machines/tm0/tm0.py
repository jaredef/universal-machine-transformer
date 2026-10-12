#!/usr/bin/env python3
"""tm0: a small deterministic single-tape transition-table executor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    next_state: str
    write: str
    move: str


@dataclass
class Program:
    blank: str
    start: str
    accept: str
    reject: str
    tape: list[str]
    rules: dict[tuple[str, str], Rule]


def parse_program(text: str) -> Program:
    header: dict[str, str] = {}
    rules: dict[tuple[str, str], Rule] = {}
    in_rules = False

    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "---":
            in_rules = True
            continue

        parts = line.split()
        if not in_rules:
            if len(parts) < 2:
                raise ValueError(f"line {line_no}: malformed header")
            key = parts[0]
            value = " ".join(parts[1:])
            if key not in {"blank", "start", "accept", "reject", "tape"}:
                raise ValueError(f"line {line_no}: unknown header {key!r}")
            header[key] = value
            continue

        if len(parts) != 6 or parts[2] != "->":
            raise ValueError(f"line {line_no}: malformed rule")
        state, read, _, next_state, write, move = parts
        if len(read) != 1:
            raise ValueError(f"line {line_no}: read must be one symbol")
        if len(write) != 1:
            raise ValueError(f"line {line_no}: write must be one symbol")
        if move not in {"L", "R", "S"}:
            raise ValueError(f"line {line_no}: invalid move {move!r}")
        if (state, read) in rules:
            raise ValueError(f"line {line_no}: duplicate rule for {state}/{read}")
        rules[(state, read)] = Rule(next_state, write, move)

    missing = {"blank", "start", "accept", "reject", "tape"} - set(header)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"missing header field(s): {names}")

    blank = header["blank"]
    if len(blank) != 1:
        raise ValueError("blank must be one symbol")

    tape = list(header["tape"])
    if any(ch.isspace() for ch in tape):
        raise ValueError("tape symbols must be non-whitespace characters")

    return Program(
        blank=blank,
        start=header["start"],
        accept=header["accept"],
        reject=header["reject"],
        tape=tape,
        rules=rules,
    )


def run(program: Program, max_steps: int, trace: bool) -> tuple[str, int, dict[int, str]]:
    tape = {i: symbol for i, symbol in enumerate(program.tape) if symbol != program.blank}
    head = 0
    state = program.start

    for step in range(max_steps + 1):
        if trace:
            print_trace(step, state, head, tape, program.blank)

        if state == program.accept:
            return ("ACCEPT", step, tape)
        if state == program.reject:
            return ("REJECT", step, tape)
        if step == max_steps:
            return ("TIMEOUT", step, tape)

        read = tape.get(head, program.blank)
        rule = program.rules.get((state, read))
        if rule is None:
            return ("REJECT", step, tape)

        if rule.write == program.blank:
            tape.pop(head, None)
        else:
            tape[head] = rule.write

        state = rule.next_state
        if rule.move == "L":
            head -= 1
        elif rule.move == "R":
            head += 1

    raise AssertionError("unreachable")


def print_trace(step: int, state: str, head: int, tape: dict[int, str], blank: str) -> None:
    keys = set(tape) | {head, 0}
    lo = min(keys)
    hi = max(keys)
    cells = "".join(tape.get(i, blank) for i in range(lo, hi + 1))
    caret = " " * (head - lo) + "^"
    print(f"step={step} state={state} head={head} tape={cells}")
    print(f"  {caret}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("program", type=Path)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    program = parse_program(args.program.read_text())
    verdict, steps, tape = run(program, args.max_steps, args.trace)
    if tape:
        lo = min(min(tape), 0)
        hi = max(max(tape), 0)
        final_tape = "".join(tape.get(i, program.blank) for i in range(lo, hi + 1))
    else:
        final_tape = program.blank
    print(f"{verdict} steps={steps} tape={final_tape}")
    return 0 if verdict == "ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
