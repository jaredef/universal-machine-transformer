#!/usr/bin/env python3
"""Milestone 5: a big program for universal machine version 2, the sieve of Eratosthenes.

The tape is the numbers themselves: cell n holds 1 while n may be prime and 0 once it is crossed out (cells 0 and 1 start
at 0). For k = 2, 3, ...: if cell k is still 1, k is prime; the machine marks it, walks right to the end of the tape
crossing out every k-th cell (2k, 3k, ...), walks back to the mark, restores it, and moves on to k + 1.

The countdown to the next multiple is held in the machine's states, one chain of k states per stride. That is what makes
the program big: sieving 0 .. N needs the strides of the primes up to sqrt(N), here 2 .. 37 for N = 1680 (41^2 = 1681),
and 247 states in all. Only primes get a stride chain: when k is composite its cell is already 0 and the machine moves on
(a chain for it would never run).

    python3 primes.py [N]       build the machine, run it directly and through version 2, and print its costs
"""
import sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fast-universal-machine"))
import utm2, check  # noqa: E402


def primes_upto(n):
    return [p for p in range(2, n + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def sieve(N=1680):
    """The machine's rules, start state and input tape, sieving the numbers 0 .. N."""
    top = int(N ** 0.5)
    base = primes_upto(top)
    rules = ["s 0 -> s1 0 R", "s1 0 -> k2 0 R"]
    for k in range(2, top + 1):
        nxt = f"k{k + 1}" if k < top else "qa"
        rules.append(f"k{k} 0 -> {nxt} 0 R")                  # crossed out already: composite
        if k not in base:
            continue
        rules.append(f"k{k} 1 -> m{k}_{k - 1} _ R")              # prime: mark it, k - 1 cells to its next multiple
        for j in range(k):
            for x in "01":
                rules.append(f"m{k}_{j} {x} -> m{k}_{(j - 1) % k} {x if j else 0} R")
            rules.append(f"m{k}_{j} _ -> r{k} _ L")              # the end of the tape: back to the mark
        rules += [f"r{k} 0 -> r{k} 0 L", f"r{k} 1 -> r{k} 1 L", f"r{k} _ -> {nxt} 1 R"]
    tape = "00" + "1" * (N - 1)
    return rules, "s", tape


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1680
    rules, start, tape = sieve(N)
    prog = check.source(rules, start, tape)
    states = utm2.states_of(prog)
    print(f"sieve to {N}: {len(states)} states, {len(rules)} rules; description {len(utm2.encode(prog, tape)) - len(tape) - 1:,} cells")
    want, verdict = check.direct(prog, tape, 10 ** 7)
    final = want[-1][2]
    got = [i for i, c in enumerate(final) if c == "1"]
    print(f"direct: {len(want):,} steps, {verdict}; primes found {len(got)}, correct: {got == primes_upto(N)}")
    t0 = time.time()
    configs, v, passes = check.through(2, prog, tape, 10 ** 7)
    ok = v == verdict and len(configs) == len(want) and all(g[0] == w[0] and g[2] == w[2] for g, w in zip(configs, want))
    print(f"version 2: {'matches every step' if ok else 'MISMATCH'}; {passes:,} forward passes on the lagging decoder, "
          f"{passes / len(want):,.0f} per described step ({time.time() - t0:.0f}s)")
