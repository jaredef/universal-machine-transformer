#!/usr/bin/env python3
"""Milestone 1: is universal machine version 2 correct, and how much faster is it?

Runs both universal machines directly (as ordinary Turing machines, on a tape that grows both ways, as the lagging-window
decoder's ring does) on many described machines, and after every step of the described machine compares its state,
head, and tape, read off the universal machine's tape, with the described machine run directly. It also measures the
cost on the lagging-window decoder exactly: every step of the universal machine is one trip around the ring, so the
forward passes for a described step are the ring lengths summed over the universal steps that make it.

    python3 check.py            correctness on every program, both versions; cost per described step
    python3 check.py --scale    cost as the number of states grows (the same computation, its states spread among more)
"""
import random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for d in ("machines/lag3/derived", "machines/lag4/derived", "machines/lag5/derived", "machines/lag0/derived"):
    sys.path.insert(0, str(ROOT / d))
import utm2  # noqa: E402
import lag3, lag0, utm as utm1, ca, bb  # noqa: E402

V = {1: (utm1, {(s, r): (n, w, m) for s, r, n, w, m in utm1.utm_rules()}),
     2: (utm2, {(s, r): (n, w, m) for s, r, n, w, m in utm2.utm_rules()})}
UNMARK = {"O": "0", "I": "1", "U": "_"}


def direct(prog, tape, limit):
    """The described machine run directly (one-way tape): its configuration after every step."""
    t = {i: s for i, s in enumerate(tape) if s != prog.blank}
    h, q, out = 0, prog.start, []
    for _ in range(limit):
        if q in (prog.accept, prog.reject):
            break
        r = prog.rules.get((q, t.get(h, prog.blank)))
        if r is None:                                        # no rule: it halts, rejecting, without a step
            q = prog.reject
            out.append((q, None, "".join(t.get(i, "_") for i in range(max(t) + 1)).rstrip("_") if t else ""))
            break
        q, w, m = r
        if w == prog.blank: t.pop(h, None)
        else: t[h] = w
        h = max(0, h + {"L": -1, "R": 1, "S": 0}[m])
        out.append((q, h, "".join(t.get(i, "_") for i in range(max(t) + 1)).rstrip("_") if t else ""))
    return out, (q if q in (prog.accept, prog.reject) else None)   # None: still running at the limit


def through(version, prog, tape, limit, order=None):
    """The described machine run by universal machine `version`: configurations after every described step (read off
    the universal machine's tape), the verdict, and the forward passes per described step on the lagging decoder."""
    mod, R = V[version]
    states = order or utm2.states_of(prog)
    t = list(mod.encode(prog, tape, order) if version == 2 else mod.encode(prog, tape))
    h, q, configs, passes, steps, prev = 0, "find", [], 0, 0, None
    while steps < limit:
        sym = t[h] if h < len(t) else "_"
        if h == len(t): t.append("_")
        r = R.get((q, sym))
        if r is None:
            break
        n, w, m = r
        t[h] = w
        passes += len(t) + 1
        if q == "fin" and n == "find":                       # a described step is complete: read the configuration
            prog_part, tape_part = "".join(t).split("$")
            k = prog_part[:prog_part.index("*")].count("|")
            cells = [UNMARK.get(c, c) for c in tape_part]
            head = next(i for i, c in enumerate(tape_part) if c in UNMARK)
            configs.append((states[k], head, "".join(cells).rstrip("_")))
            steps += 1
        h += {"L": -1, "R": 1, "S": 0}[m]
        q = n
        if h < 0:
            t.insert(0, "_"); h = 0
        if q in ("qa", "qr"):
            break
    if steps >= limit:
        return configs, None, passes
    verdict = "qa" if q == "qa" else "qr"
    if True:                                                 # halted: record the final configuration
        prog_part, tape_part = "".join(t).split("$")
        configs.append((prog.accept if verdict == "qa" else prog.reject, None, "".join(UNMARK.get(c, c) for c in tape_part).rstrip("_")))
    return configs, verdict, passes


def source(rules, start, tape):
    return lag0.parse_tm0(f"blank _\nstart {start}\naccept qa\nreject qr\ntape {tape}\n---\n" + "\n".join(rules) + "\n")


def programs():
    P = []
    for name, tape in __import__("lag3w").suite():
        P.append((f"{name}({tape})", lag0.parse_tm0(lag3.held_out_source(name, tape)), tape))
    dragon = ["s _ -> inc _ R", "inc 1 -> inc 0 R", "inc 0 -> back 1 L", "inc _ -> qa _ S", "back 0 -> back 0 L", "back 1 -> back 1 L",
              "back _ -> low _ R", "low 0 -> low 0 R", "low 1 -> nxt 1 R", "nxt 0 -> sayR 0 L", "nxt _ -> sayR _ L", "nxt 1 -> sayL 1 L",
              "sayR 1 -> home 1 L", "sayL 1 -> home 1 L", "home 0 -> home 0 L", "home _ -> inc _ R"]
    P.append(("dragon (8 bits)", source(dragon, "s", "_" + "0" * 8), "_" + "0" * 8))
    fib = ["s _ -> A0 _ R"]
    for c in (0, 1):
        for x in (0, 1): fib.append(f"A{c} {x} -> B{c}{x} {x} R")
        for x in (0, 1):
            for y in (0, 1): s = x + y + c; fib.append(f"B{c}{x} {y} -> W{y}{s >> 1} {s & 1} L")
    for y in (0, 1):
        for c in (0, 1):
            for z in (0, 1): fib.append(f"W{y}{c} {z} -> N{c} {y} R")
    for c in (0, 1):
        for z in (0, 1): fib.append(f"N{c} {z} -> A{c} {z} R")
    fib += ["A0 _ -> home _ L", "A1 _ -> qa _ S", "home 0 -> home 0 L", "home 1 -> home 1 L", "home _ -> A0 _ R"]
    P.append(("fibonacci (8 bits)", source(fib, "s", "_01" + "00" * 7), "_01" + "00" * 7))
    collatz = ["s _ -> s _ R", "s 0 -> s _ R", "s 1 -> one 0 R", "one _ -> fix _ L", "one 0 -> c1 0 R", "one 1 -> c2 1 R", "fix 0 -> qa 1 S",
               "c0 0 -> c0 0 R", "c0 1 -> c1 1 R", "c0 _ -> back _ L", "c1 0 -> c0 1 R", "c1 1 -> c2 0 R", "c1 _ -> back 1 L",
               "c2 0 -> c1 0 R", "c2 1 -> c2 1 R", "c2 _ -> c1 0 R", "back 0 -> back 0 L", "back 1 -> back 1 L", "back _ -> s _ R"]
    P.append(("collatz(27)", source(collatz, "s", "_11011"), "_11011"))
    for rule in (90, 30, 110):
        P.append((f"rule {rule}", lag0.parse_tm0(ca.ca_source(rule)), "_1"))
    for name in ("bb2", "bb4"):
        src = bb.bb_source(name)
        P.append((name, lag0.parse_tm0(src), "0" * bb.pad(name)))
    rng = random.Random(7)
    for i in range(300):                                     # random machines, as in lag3's training data
        src = lag3.random_source(rng, rng.randint(1, 5), "".join(rng.choice("01_") for _ in range(rng.randint(1, 6))),
                                 "RRRLS" if i % 2 else "LRS", 0.5)
        p = lag0.parse_tm0(src)
        P.append((f"random {i}", p, "".join(p.tape)))
    return P


def main():
    if "--scale" in sys.argv:
        return scale()
    t0, bad, rows = time.time(), 0, []
    for name, prog, tape in programs():
        want, wq = direct(prog, tape, 3000)
        res = {}
        for v in (1, 2):
            got, verdict, passes = through(v, prog, tape, 3000)
            ok = verdict == wq and len(got) == len(want) and all(g[0] == w[0] and g[2] == w[2] and (g[1] is None or w[1] is None or g[1] == w[1]) for g, w in zip(got, want))
            res[v] = (ok, passes / max(1, len(got)))
            bad += not ok
        rows.append((name, len(want), res))
        if not (res[1][0] and res[2][0]):
            print(f"MISMATCH {name}: v1 {res[1][0]}, v2 {res[2][0]}", flush=True)
    named = [r for r in rows if not r[0].startswith("random")]
    print(f"{len(rows)} programs ({len(rows) - len(named)} random), {sum(r[1] for r in rows):,} described steps, every configuration compared: "
          f"{'all match, both versions' if not bad else str(bad) + ' MISMATCHES'}   ({time.time() - t0:.0f}s)")
    print("\n| Program | Described steps | Forward passes per step, v1 | v2 | Speedup |\n|---|---|---|---|---|")
    for name, n, res in named:
        if name.split("(")[0] in ("increment", "palindrome") or not any(c.isdigit() for c in name.split("(")[-1][:1]) or name.startswith(("dragon", "fib", "collatz", "rule", "bb")):
            pass
    shown = set()
    for name, n, res in named:
        key = name.split("(")[0]
        if key in shown and key not in ("rule 90", "rule 30", "rule 110"):
            continue
        shown.add(key)
        print(f"| {name} | {n:,} | {res[1][1]:,.0f} | {res[2][1]:,.0f} | {res[1][1] / res[2][1]:.1f}x |")
    return 0 if not bad else 1


def scale():
    """The dragon-curve computation (8 states) with its states spread among more, at random places: the work is the
    same, only the program gets bigger. Above 32 states, version 1 is measured over its first few described steps only."""
    dragon = ["s _ -> inc _ R", "inc 1 -> inc 0 R", "inc 0 -> back 1 L", "inc _ -> qa _ S", "back 0 -> back 0 L", "back 1 -> back 1 L",
              "back _ -> low _ R", "low 0 -> low 0 R", "low 1 -> nxt 1 R", "nxt 0 -> sayR 0 L", "nxt _ -> sayR _ L", "nxt 1 -> sayL 1 L",
              "sayR 1 -> home 1 L", "sayL 1 -> home 1 L", "home 0 -> home 0 L", "home _ -> inc _ R"]
    print("| States | v1 passes per step | v2 passes per step | Speedup | v2 correct |\n|---|---|---|---|---|")
    for S in (8, 16, 32, 64, 128, 256):
        rng = random.Random(S)
        pad = [f"z{i}" for i in range(S - 8)]                  # states never reached, each with one rule so they are encoded
        rules = dragon + [f"{z} 0 -> {z} 0 R" for z in pad]
        prog = source(rules, "s", "_" + "0" * 6)
        real = utm2.states_of(prog)
        rest = real[1:]; rng.shuffle(rest); order = [real[0]] + rest
        want, _ = direct(prog, "_" + "0" * 6, 10 ** 6)
        got2, _, p2 = through(2, prog, "_" + "0" * 6, 10 ** 6, order)
        ok2 = len(got2) == len(want) and all(g[0] == w[0] and g[2] == w[2] for g, w in zip(got2, want))
        lim = {64: 20, 128: 3, 256: 1}.get(S, 10 ** 6)          # version 1 gets slow to simulate: its first few steps only
        got1, _, p1 = through(1, reorder(prog, order), "_" + "0" * 6, lim)
        print(f"| {S} | {p1 / len(got1):,.0f} | {p2 / len(got2):,.0f} | {p1 / len(got1) / (p2 / len(got2)):,.0f}x | {'yes' if ok2 else 'NO'} |", flush=True)


def reorder(prog, order):
    """The same machine with its states renamed so that version 1's order (start first, then sorted) is `order`."""
    names = {q: f"q{i:03d}" for i, q in enumerate(order)}
    names[prog.accept], names[prog.reject] = prog.accept, prog.reject
    lines = [f"{names[q]} {s} -> {names[n]} {w} {m}" for (q, s), (n, w, m) in prog.rules.items()]
    return source(lines, names[order[0]], "".join(prog.tape))


if __name__ == "__main__":
    sys.exit(main())
