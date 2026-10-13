#!/usr/bin/env python3
"""Does a narrower network generalize better? (needs torch)

The smallest-network experiment found that the narrower the network, the more its head-token embeddings are
"state + symbol": squeezing forces it to learn the structure the tokens hide. If that structure is what generalization
needs, narrow networks should run unseen programs from fewer random machines than wide ones.

For each width and each training set (the steps of the first N of lag3's random machines, as in
../smallest-training-set), train with one recipe for every width (full-batch Adam, learning rate 0.003, stopping when
every training step is right, at most 20,000 steps), then score the network on every step it never saw, and on the
twelve demonstration programs: a program runs correctly when every step it needs is right.

    python train.py                       the grid below, seeds 0-2; writes results.json
    SEEDS=3,4 python train.py 8:2000      chosen width:N pairs
    FIXED=1500 python train.py 8:500 ...  the control: exactly 1,500 Adam steps for every width, no stopping early;
                                          writes results_fixed1500.json
"""
import json, math, os, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
S = json.loads((HERE.parent / "smallest-training-set" / "steps.json").read_text())
steps = [s.split("\u0001") for s in S["steps"]]
used_in, used_out = {t for w in steps for t in w}, set(S["labels"])
VIN = [t for t in S["vin"] if t in used_in]
VOUT = [t for t in S["vout"] if t in used_out]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
X = torch.tensor([[VIN.index(t) for t in w] for w in steps], device=DEV)
Y = torch.tensor([VOUT.index(y) for y in S["labels"]], device=DEV)
FIRST = torch.tensor(S["first"], device=DEV)
TESTS = {k: torch.tensor(v, device=DEV) for k, v in S["tests"].items()}
NEEDED = torch.tensor(sorted(set().union(*map(set, S["tests"].values()))), device=DEV)

WIDTHS, MACHINES = [8, 16, 64], [200, 500, 1000, 2000, 2500]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
LR, MAX_STEPS = 0.003, 20000
FIXED = int(os.environ.get("FIXED", "0"))                   # the control: train every width for exactly this many steps


class Model(torch.nn.Module):                            # lag3's architecture at width d, MLP 4d
    def __init__(s, d, heads=2):
        super().__init__()
        s.d, s.h = d, heads
        s.E = torch.nn.Embedding(len(VIN), d); s.P = torch.nn.Parameter(torch.randn(3, d) * 0.1)
        s.Wq, s.Wk, s.Wv, s.Wo = (torch.nn.Linear(d, d, bias=False) for _ in range(4))
        s.W1, s.W2 = torch.nn.Linear(d, 4 * d), torch.nn.Linear(4 * d, d)
        s.U = torch.nn.Linear(d, len(VOUT))

    def forward(s, idx):
        x = s.E(idx) + s.P
        n, t, d = x.shape; dh = d // s.h
        q = s.Wq(x[:, -1:]).view(n, 1, s.h, dh).transpose(1, 2)
        k = s.Wk(x).view(n, t, s.h, dh).transpose(1, 2)
        v = s.Wv(x).view(n, t, s.h, dh).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
        y = (a @ v).transpose(1, 2).reshape(n, 1, d)
        h = x[:, -1:] + s.Wo(y)
        h = h + s.W2(torch.relu(s.W1(h)))
        return s.U(h[:, 0])


def run(d, n, seed):
    inside = (FIRST >= 0) & (FIRST < n)
    tr = inside.nonzero().squeeze(1)
    torch.manual_seed(seed)
    m = Model(d).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=LR)
    fit_at, t0 = None, time.time()
    for step in range(1, (FIXED or MAX_STEPS) + 1):
        loss = torch.nn.functional.cross_entropy(m(X[tr]), Y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        if not FIXED and step % 50 == 0:
            with torch.no_grad():
                if bool((m(X[tr]).argmax(-1) == Y[tr]).all()):
                    fit_at = step; break
    with torch.no_grad():
        right = m(X).argmax(-1) == Y
    unseen = ~inside
    need_unseen = unseen[NEEDED]
    return {"d": d, "n": n, "seed": seed, "fixed_steps": FIXED or None, "params": sum(p.numel() for p in m.parameters()), "train_steps": len(tr),
            "fit_at": fit_at, "train_errors": int((~right[inside]).sum()),
            "unseen": int(unseen.sum()), "unseen_right": int(right[unseen].sum()),
            "needed_unseen": int(need_unseen.sum()), "needed_unseen_right": int(right[NEEDED][need_unseen].sum()),
            "programs": sum(bool(right[idx].all()) for idx in TESTS.values()), "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    grid = [tuple(int(x) for x in a.split(":")) for a in sys.argv[1:]] or [(d, n) for n in MACHINES for d in WIDTHS]
    path = HERE / (f"results_fixed{FIXED}.json" if FIXED else "results.json")
    results = json.loads(path.read_text()) if path.exists() else {"runs": []}
    for d, n in grid:
        for seed in SEEDS:
            r = run(d, n, seed)
            results["runs"] = [x for x in results["runs"] if (x["d"], x["n"], x["seed"]) != (d, n, seed)] + [r]
            path.write_text(json.dumps(results, indent=1))
            print(f"width {d:>2} ({r['params']:>6,} params) N={n:>5} seed={seed}: fit at {r['fit_at']}; "
                  f"unseen steps right {r['unseen_right']}/{r['unseen']}; the programs' unseen steps right "
                  f"{r['needed_unseen_right']}/{r['needed_unseen']}; programs correct {r['programs']}/12 ({r['seconds']}s)", flush=True)
