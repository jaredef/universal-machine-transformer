#!/usr/bin/env python3
"""lag1: lag0's rule table realized as transformer weights.

lag0 runs a Turing machine on an autoregressive loop whose three-token window lags behind the frontier, with the
step given by a lookup table. lag1 replaces the lookup table with a transformer forward pass:

    embedding      each token -> its features (boundary, head, tape symbol, state) + a position one-hot
    attention      two heads with hard attention, read at the last position of the window:
                     the previous-token head copies the middle token, the two-back head copies the left one
    MLP 1          the Turing rule table as neurons: one ReLU unit per row (q, a) of delta, per slot; it fires
                   when the slot holds state q reading a, and writes that row's move, next state, and symbol
    MLP 2          ReLU units that are Boolean ANDs of those features, assembling the output token's features
    unembedding    logit(v) = 2 <f, u_v> - |u_v|: maximal, and unique, exactly when f equals v's features

Every weight is a small integer and every activation is exact (attention averages use Fractions), so the
claims below are checked exactly, not approximately:

    LT-1  the forward pass equals lag0's rule table on every window the decoder can meet (exhaustive)
    LT-2  decoding with the forward pass agrees with tm0 on every fixture: verdict, steps, and tape
    LT-3  the same weights under a recency window never compute
    LT-4  MLP 1 has exactly one neuron per row of the Turing rule table, per slot

The weights are derived from the .tm0 program, not trained: an existence proof by construction.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lag0" / "derived"))
import lag0  # noqa: E402  (the reference: parse_tm0, rule, alphabet, read_tape, initial_ring)

B, SEP = lag0.BOUNDARY, lag0.SEP
HALT = "<halt>"


class Net:
    """A two-layer transformer whose weights are derived from a .tm0 program."""

    def __init__(self, prog: lag0.Program):
        self.prog = prog
        syms, states = set(), set()
        for t in lag0.alphabet(prog):
            h = lag0.split_head(t)
            if h:
                states.add(h[0]); syms.add(h[1])
            elif t != B:
                syms.add(t)
        self.syms, self.states = sorted(syms), sorted(states)
        self.vocab_in = lag0.alphabet(prog)

        # ---- the residual stream: named dimensions
        tok = ["isB", "isH"] + [f"sym:{a}" for a in self.syms] + [f"st:{q}" for q in self.states]
        der = ["has", "mvL", "mvR", "mvS"] + [f"nq:{q}" for q in self.states] + [f"wr:{a}" for a in self.syms]
        out = ["oB", "oH", "halt"] + [f"oSt:{q}" for q in self.states] + [f"oSym:{a}" for a in self.syms] \
            + [f"grow:{q}" for q in self.states]
        names = [f"r.{f}" for f in tok] + ["pos0", "pos1", "pos2"] + [f"c.{f}" for f in tok] + [f"l.{f}" for f in tok]
        names += [f"{s}.{f}" for s in "lcr" for f in der] + [f"F.{f}" for f in out]
        self.dims = names
        self.ix = {n: i for i, n in enumerate(names)}
        self.tokf, self.outf = tok, out
        d = self.d = len(names)
        z = lambda r, c: [[0] * c for _ in range(r)]

        # ---- embedding and positions
        self.W_E = {}
        for t in self.vocab_in:
            v = [0] * d
            h = lag0.split_head(t)
            if t == B:
                v[self.ix["r.isB"]] = 1
            elif h:
                v[self.ix["r.isH"]] = 1; v[self.ix[f"r.sym:{h[1]}"]] = 1; v[self.ix[f"r.st:{h[0]}"]] = 1
            else:
                v[self.ix[f"r.sym:{t}"]] = 1
            self.W_E[t] = v
        self.P = [[1 if j == self.ix[f"pos{i}"] else 0 for j in range(d)] for i in range(3)]

        # ---- attention: two heads, keys and queries in position space (d_k = 3), values copy the token slot
        self.heads = []
        for back, slot in ((1, "c"), (2, "l")):
            W_Q, W_K = z(d, 3), z(d, 3)
            for i in range(3):
                W_K[self.ix[f"pos{i}"]][i] = 1
                if i - back >= 0:
                    W_Q[self.ix[f"pos{i}"]][i - back] = 1       # position i looks for position i - back
            W_V, W_O = z(d, len(tok)), z(len(tok), d)
            for k, f in enumerate(tok):
                W_V[self.ix[f"r.{f}"]][k] = 1
                W_O[k][self.ix[f"{slot}.{f}"]] = 1
            self.heads.append((W_Q, W_K, W_V, W_O))

        # ---- MLP 1: the Turing rule table as neurons
        units1 = []                                              # (input weights {dim: w}, bias, output {dim: w})
        for s in "lcr":
            src = "r" if s == "r" else s
            for (q, a), (nq, wr, mv) in sorted(prog.rules.items()):
                if q in (prog.accept, prog.reject):
                    continue
                units1.append(({f"{src}.st:{q}": 1, f"{src}.sym:{a}": 1}, -1,
                               {f"{s}.has": 1, f"{s}.mv{mv}": 1, f"{s}.nq:{nq}": 1, f"{s}.wr:{wr}": 1}))
        self.mlp1 = units1
        self.delta_rows = sum(1 for (q, _) in prog.rules if q not in (prog.accept, prog.reject))

        # ---- MLP 2: assemble the output token's features (each unit is a Boolean AND)
        u = []
        notplain = {"c.isH": -1, "c.isB": -1}                  # 1 - isH - isB, folded into each AND below
        u.append(({"c.isB": 1}, 0, {"F.oB": 1}))
        u.append(({"c.isH": 1, "c.has": -1}, 0, {"F.halt": 1}))
        u.append(({"c.mvS": 1}, 0, {"F.oH": 1}))
        u.append(({"l.mvR": 1, **notplain}, 0, {"F.oH": 1}))
        u.append(({"r.mvL": 1, **notplain}, 0, {"F.oH": 1}))
        for q in self.states:
            u.append(({"c.mvS": 1, f"c.nq:{q}": 1}, -1, {f"F.oSt:{q}": 1}))
            u.append(({"l.mvR": 1, f"l.nq:{q}": 1, **notplain}, -1, {f"F.oSt:{q}": 1}))
            u.append(({"r.mvL": 1, f"r.nq:{q}": 1, **notplain}, -1, {f"F.oSt:{q}": 1}))
            u.append(({"c.isB": 1, "r.mvL": 1, f"r.nq:{q}": 1}, -2, {f"F.grow:{q}": 1}))
            u.append(({"c.mvR": 1, f"c.nq:{q}": 1, "r.isB": 1}, -2, {f"F.grow:{q}": 1}))
        for a in self.syms:
            u.append(({f"c.wr:{a}": 1}, 0, {f"F.oSym:{a}": 1}))
            u.append(({f"c.sym:{a}": 1, **notplain}, 0, {f"F.oSym:{a}": 1}))
        self.mlp2 = u

        # ---- unembedding over the output vocabulary
        self.vocab_out: list[tuple[str, list[str], set[str]]] = []  # (name, transcript tokens, feature set)
        blank = prog.blank
        self.vocab_out.append((B, [B], {"oB"}))
        self.vocab_out.append((HALT, [], {"halt"}))
        for a in self.syms:
            self.vocab_out.append((a, [a], {f"oSym:{a}"}))
            for q in self.states:
                self.vocab_out.append((lag0.head(q, a), [lag0.head(q, a)], {"oH", f"oSt:{q}", f"oSym:{a}"}))
                self.vocab_out.append((f"{a}|{q}:{blank}", [a, lag0.head(q, blank)], {f"oSym:{a}", f"grow:{q}"}))
        for q in self.states:
            self.vocab_out.append((f"#|{q}:{blank}", [B, lag0.head(q, blank)], {"oB", f"grow:{q}"}))
        self.W_U = [[2 if f"F.{n}" in {f"F.{x}" for x in feats} else 0 for n in self.outf] for _, _, feats in self.vocab_out]
        self.b_U = [-len(feats) for _, _, feats in self.vocab_out]

    # ---------------------------------------------------------------- the forward pass

    def mlp(self, x, units):
        delta = [0] * self.d
        for w_in, bias, w_out in units:
            a = bias + sum(c * x[self.ix[n]] for n, c in w_in.items())
            if a > 0:
                for n, c in w_out.items():
                    delta[self.ix[n]] += c * a
        return [xi + di for xi, di in zip(x, delta)]

    def forward(self, window: list[str]) -> tuple[str, list[str]]:
        n = len(window)
        X = [[e + p for e, p in zip(self.W_E[t], self.P[i])] for i, t in enumerate(window)]
        # attention: causal, hard (argmax), exact averages
        out = [row[:] for row in X]
        for W_Q, W_K, W_V, W_O in self.heads:
            Q = [mat_vec_T(W_Q, x) for x in X]
            K = [mat_vec_T(W_K, x) for x in X]
            V = [mat_vec_T(W_V, x) for x in X]
            for i in range(n):
                scores = [sum(a * b for a, b in zip(Q[i], K[j])) for j in range(i + 1)]
                top = max(scores)
                picked = [j for j in range(i + 1) if scores[j] == top]
                avg = [Fraction(sum(V[j][k] for j in picked), len(picked)) for k in range(len(V[0]))]
                w = mat_vec_T(W_O, avg)
                out[i] = [a + b for a, b in zip(out[i], w)]
        x = out[n - 1]                                   # the prediction is read at the last position
        x = self.mlp(x, self.mlp1)
        x = self.mlp(x, self.mlp2)
        f = [x[self.ix[f"F.{n_}"]] for n_ in self.outf]
        logits = [sum(w * fi for w, fi in zip(row, f)) + b for row, b in zip(self.W_U, self.b_U)]
        best = max(range(len(logits)), key=lambda k: logits[k])
        if sorted(logits)[-1] == sorted(logits)[-2]:
            raise AssertionError(f"tied logits on window {window}")
        name, emits, _ = self.vocab_out[best]
        return name, emits

    def explain(self, window: list[str]) -> list[str]:
        """The forward pass on one window, layer by layer, in words."""
        lines = [f"window   [{' '.join(window)}]   read at position 2 ({window[2]})"]
        X = [[e + p for e, p in zip(self.W_E[t], self.P[i])] for i, t in enumerate(window)]
        x = X[2][:]
        for (W_Q, W_K, W_V, W_O), label in zip(self.heads, ("previous-token head", "two-back head")):
            q = mat_vec_T(W_Q, X[2]); scores = [sum(a * b for a, b in zip(q, mat_vec_T(W_K, X[j]))) for j in range(3)]
            j = max(range(3), key=lambda k: scores[k])
            lines.append(f"attn     {label}: scores {scores} -> reads position {j} ({window[j]})")
            w = mat_vec_T(W_O, mat_vec_T(W_V, X[j]))
            x = [a + b for a, b in zip(x, w)]
        for units, label in ((self.mlp1, "mlp1"), (self.mlp2, "mlp2")):
            fired = []
            for w_in, bias, w_out in units:
                a = bias + sum(c * x[self.ix[n]] for n, c in w_in.items())
                if a > 0:
                    ins = " + ".join(n for n in w_in if x[self.ix[n]] and w_in[n] > 0)
                    fired.append(f"[{ins}] -> {', '.join(w_out)}")
            x = self.mlp(x, units)
            lines += [f"{label}     {f}" for f in fired] or [f"{label}     (no unit fires)"]
        f = [x[self.ix[f"F.{n_}"]] for n_ in self.outf]
        logits = [sum(w * fi for w, fi in zip(row, f)) + b for row, b in zip(self.W_U, self.b_U)]
        order = sorted(range(len(logits)), key=lambda k: -logits[k])[:3]
        feats = [n_ for n_, v in zip(self.outf, f) if v]
        lines.append(f"features {feats}")
        lines.append("logits   " + ", ".join(f"{self.vocab_out[k][0]}={logits[k]}" for k in order))
        lines.append(f"emit     {self.vocab_out[order[0]][1] or '<halt>'}")
        return lines

    def parameters(self) -> dict[str, int]:
        nz = lambda m: sum(1 for row in m for v in row if v)
        attn = sum(nz(q) + nz(k) + nz(v) + nz(o) for q, k, v, o in self.heads)
        m1 = sum(len(i) + 1 + len(o) for i, _, o in self.mlp1)
        m2 = sum(len(i) + 1 + len(o) for i, _, o in self.mlp2)
        return {"d_model": self.d, "attention": attn, "mlp1_units": len(self.mlp1), "mlp2_units": len(self.mlp2),
                "mlp_weights": m1 + m2, "embedding": sum(nz([v]) for v in self.W_E.values()) + 3,
                "unembedding": nz(self.W_U), "vocab_in": len(self.vocab_in), "vocab_out": len(self.vocab_out)}


def mat_vec_T(W, x):
    """x (length rows of W) times W -> length cols of W."""
    cols = len(W[0])
    return [sum(x[r] * W[r][c] for r in range(len(W)) if x[r]) for c in range(cols)]


# ---------------------------------------------------------------- the decoder


def is_reachable(window) -> bool:
    """At most one head, except around '#' in a one-cell ring: there the window sees the same cell from two
    generations, so the left and right of '#' can both be heads (the old one and the new one)."""
    l, c, r = window
    heads = [lag0.split_head(t) is not None for t in (l, c, r)]
    return sum(heads) <= 1 or (heads == [True, False, True] and c == B)


def reachable_windows(prog: lag0.Program):
    """Every window the lagging decoder can meet. LT-1 checks the forward pass on all of them; decode() asserts
    that every window it actually meets is one of them."""
    A = lag0.alphabet(prog)
    for l in A:
        for c in A:
            for r in A:
                if is_reachable((l, c, r)):
                    yield (l, c, r)


def decode(prog: lag0.Program, net: Net, discipline: str = "lag", max_steps: int = 50_000):
    ring = lag0.initial_ring(prog)
    transcript, k, tm_steps, left_grown, seen = [ring[-1], *ring], 0, 0, 0, set()
    domain = None
    for step in range(max_steps):
        window = transcript[k:k + 3] if discipline == "lag" else transcript[-3:]
        if discipline == "lag":
            assert is_reachable(window), f"window outside the checked domain: {window}"
        name, emits = net.forward(window)
        if not emits:                                     # the <halt> token: no rule
            if discipline == "recency":
                return "NO-RULE", tm_steps, step, ""
            h = lag0.split_head(window[1])
            verdict = "ACCEPT" if h and h[0] == prog.accept else "REJECT"
            return verdict, tm_steps, step, lag0.read_tape(prog, transcript[k + 1:], left_grown)
        if discipline == "recency":
            if tuple(window) in seen:
                return "CYCLE", tm_steps, step, ""
            seen.add(tuple(window))
        if lag0.step_of(prog, window[1]):
            tm_steps += 1
        if window[1] == B and len(emits) == 2:
            left_grown += 1
        transcript.extend(emits)
        k += 1
    return "TIMEOUT", tm_steps, max_steps, ""


# ---------------------------------------------------------------- probes


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: lag1.py {exhaustive|decode|recency|neurons|params} PROGRAM.tm0 | lag1.py explain PROGRAM.tm0 'L C R'")
        return 2
    mode, path = sys.argv[1], Path(sys.argv[2])
    prog = lag0.parse_tm0(path.read_text())
    net = Net(prog)
    if mode == "exhaustive":
        n = bad = 0
        for w in reachable_windows(prog):
            want = lag0.rule(prog, *w)
            _, got = net.forward(list(w))
            n += 1
            if (want or []) != got:
                bad += 1
                if bad <= 3:
                    print(f"MISMATCH window={w} table={want} net={got}", file=sys.stderr)
        print(f"{'FORWARD-EQUALS-TABLE' if bad == 0 else 'FORWARD-DIFFERS'} windows={n} mismatches={bad}")
        return 0 if bad == 0 else 1
    if mode in ("decode", "recency"):
        verdict, steps, dsteps, tape = decode(prog, net, "lag" if mode == "decode" else "recency")
        print(f"{verdict} steps={steps} tape={tape}" if mode == "decode" else f"{verdict} decode_steps={dsteps}")
        return 0
    if mode == "neurons":
        per_slot = len(net.mlp1) // 3
        ok = per_slot == net.delta_rows
        print(f"{'RULE-TABLE-IS-NEURONS' if ok else 'NEURON-COUNT-DIFFERS'} delta_rows={net.delta_rows} "
              f"mlp1_units={len(net.mlp1)} per_slot={per_slot}")
        return 0 if ok else 1
    if mode == "explain":
        window = sys.argv[3].split()
        print("\n".join(net.explain(window)))
        return 0
    if mode == "params":
        print(" ".join(f"{k}={v}" for k, v in net.parameters().items()))
        return 0
    print(f"unknown mode {mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
