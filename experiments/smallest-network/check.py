#!/usr/bin/env python3
"""Check a trained network from fixtures/ exactly, in pure Python (standard library only).

Recomputes the network's forward pass from its committed weights on every one of the universal machine's 14,849 table
steps and compares each prediction with the exact output. Since the lagging-window decoder is deterministic, a network
that is right on every table step runs every program whose steps lie in the table exactly; the twelve demonstration
programs' steps all do (checked here too).

    python3 check.py fixtures/net_d16_h64_seed0.json [--full-vocab out.json]

--full-vocab writes the weights over data.js's full vocabularies (unused tokens: zero embedding, unused outputs: bias
-1e4), the format web/core.js and web/tests/programs-check.js run.
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE.parent / "smallest-training-set" / "steps.json").read_text())


def forward(w, d, heads, ids):
    E, P = w["E.weight"], w["P"]
    lin = lambda W, x, b=None: [sum(a * c for a, c in zip(row, x)) + (b[i] if b else 0.0) for i, row in enumerate(W)]
    X = [[e + p for e, p in zip(E[t], P[i])] for i, t in enumerate(ids)]
    dh = d // heads
    q = lin(w["Wq.weight"], X[-1]); K = [lin(w["Wk.weight"], x) for x in X]; V = [lin(w["Wv.weight"], x) for x in X]
    y = [0.0] * d
    for hd in range(heads):
        lo, hi = hd * dh, (hd + 1) * dh
        sc = [sum(a * b for a, b in zip(q[lo:hi], k[lo:hi])) / math.sqrt(dh) for k in K]
        m = max(sc); ex = [math.exp(v - m) for v in sc]; z = sum(ex)
        for j in range(3):
            for c in range(lo, hi):
                y[c] += ex[j] / z * V[j][c]
    h = [a + b for a, b in zip(X[-1], lin(w["Wo.weight"], y))]
    hid = [max(0.0, v) for v in lin(w["W1.weight"], h, w["W1.bias"][0])]
    h = [a + b for a, b in zip(h, lin(w["W2.weight"], hid, w["W2.bias"][0]))]
    z = lin(w["U.weight"], h, w["U.bias"][0])
    return max(range(len(z)), key=z.__getitem__)


def main() -> int:
    F = json.loads(Path(sys.argv[1]).read_text())
    w, vin, vout = F["weights"], F["vin"], F["vout"]
    iin = {t: i for i, t in enumerate(vin)}
    params = sum(len(m) * len(m[0]) for m in w.values())
    wrong = []
    for k, (s, lab) in enumerate(zip(S["steps"], S["labels"])):
        ids = [iin[t] for t in s.split("\u0001")]
        if vout[forward(w, F["d"], F["heads"], ids)] != lab:
            wrong.append(k)
    ok_programs = sum(all(k not in set(wrong) for k in v) for v in S["tests"].values())
    print(f"{sys.argv[1]}: width {F['d']}, MLP {F['hidden']}, {params:,} parameters; "
          f"{len(S['steps']) - len(wrong):,} of {len(S['steps']):,} table steps exactly right; "
          f"programs whose every step is right: {ok_programs} of {len(S['tests'])}")
    if "--full-vocab" in sys.argv:
        full_in, full_out = S["vin"], S["vout"]
        E = [[0.0] * F["d"] for _ in full_in]; U = [[0.0] * F["d"] for _ in full_out]; bU = [-1e4] * len(full_out)
        for i, t in enumerate(vin): E[full_in.index(t)] = w["E.weight"][i]
        for i, t in enumerate(vout): U[full_out.index(t)] = w["U.weight"][i]; bU[full_out.index(t)] = w["U.bias"][0][i]
        out = dict(w); out["E.weight"], out["U.weight"], out["U.bias"] = E, U, [bU]
        Path(sys.argv[sys.argv.index("--full-vocab") + 1]).write_text(json.dumps({"heads": F["heads"], "weights": out}))
    return 0 if not wrong else 1


if __name__ == "__main__":
    sys.exit(main())
