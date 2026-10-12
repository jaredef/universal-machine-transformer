#!/usr/bin/env python3
"""Train lag3's network on the steps of the first N random machines, for a range of N, and test it (needs torch).

Same architecture, optimizer, and step count as machines/lag3/fixtures/train_lag3.ref.py; the training set for N is
every step that the first N of lag3's random machines produce (N = 4,000 is lag3's training set exactly). For each N
and seed, it records the fit, and for each test program: how many of its steps are missing from training, how many of
its steps the network gets wrong (in and out of training), and whether the run is correct (no step wrong). It also
scores the network on the pool's steps that first appear after machine N: steps never seen, from the same distribution.

    python train.py [N ...]            writes results.json (merging with any earlier results)
"""
import json, math, os, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "steps.json").read_text())
VIN, VOUT = D["vin"], D["vout"]
SEEDS, STEPS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")], 3000   # SEEDS=3,4,5 python train.py N
GRID = [int(a) for a in sys.argv[1:]] or [10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000]
torch.set_num_threads(4)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")   # the Apple GPU when present

X_all = torch.tensor([[VIN.index(t) for t in s.split("\u0001")] for s in D["steps"]]).to(DEV)
Y_all = torch.tensor([VOUT.index(y) for y in D["labels"]]).to(DEV)
FIRST = torch.tensor(D["first"]).to(DEV)
TESTS = {k: torch.tensor(v).to(DEV) for k, v in D["tests"].items()}
NEEDED = torch.tensor(sorted(set().union(*map(set, D["tests"].values())))).to(DEV)


class Model(torch.nn.Module):                            # identical to train_lag3.ref.py
    def __init__(s, d=64, heads=2, hidden=256):
        super().__init__()
        s.d, s.h = d, heads
        s.E = torch.nn.Embedding(len(VIN), d); s.P = torch.nn.Parameter(torch.randn(3, d) * 0.1)
        s.Wq, s.Wk, s.Wv, s.Wo = (torch.nn.Linear(d, d, bias=False) for _ in range(4))
        s.W1, s.W2 = torch.nn.Linear(d, hidden), torch.nn.Linear(hidden, d)
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


def train(n, seed):
    tr = ((FIRST >= 0) & (FIRST < n)).nonzero().squeeze(1)
    X, Y = X_all[tr], Y_all[tr]
    torch.manual_seed(seed)
    m = Model().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    fit_at = None
    for step in range(STEPS + 1):
        logits = m(X)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        if step % 100 == 0 and fit_at is None and (logits.argmax(-1) == Y).all():
            fit_at = step
        if step < STEPS:
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        right = m(X_all).argmax(-1) == Y_all
    in_train = (FIRST >= 0) & (FIRST < n)
    later = FIRST >= n                                   # pool steps that first appear after machine n
    rec = {"n": n, "seed": seed, "device": DEV.type, "train_steps": len(tr), "fit_at": fit_at,
           "train_errors": int((~right[in_train]).sum()),
           "later_steps": int(later.sum()), "later_correct": int(right[later].sum()), "tests": {}}
    rec["wrong_needed"] = [int(i) for i in NEEDED[~right[NEEDED]].tolist()]   # step indices a demo needs and gets wrong
    for name, idx in TESTS.items():
        unseen = ~in_train[idx]
        rec["tests"][name] = {"needed": len(idx), "unseen": int(unseen.sum()),
                              "wrong": int((~right[idx]).sum()), "wrong_unseen": int((~right[idx] & unseen).sum()),
                              "correct_run": bool(right[idx].all())}
    return rec


if __name__ == "__main__":
    path = HERE / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {"runs": []}
    t0 = time.time()
    for n in GRID:
        for seed in SEEDS:
            r = train(n, seed)
            results["runs"] = [x for x in results["runs"] if (x["n"], x["seed"]) != (n, seed)] + [r]
            ok = sum(t["correct_run"] for t in r["tests"].values())
            uns = sum(t["unseen"] for t in r["tests"].values())
            wu = sum(t["wrong_unseen"] for t in r["tests"].values())
            print(f"N={n:>5} seed={seed} train_steps={r['train_steps']:>6} fit_at={r['fit_at']} train_errors={r['train_errors']}"
                  f" | programs correct {ok}/{len(r['tests'])} | test steps unseen {uns}, of them wrong {wu}"
                  f" | later pool steps correct {r['later_correct']}/{r['later_steps']} ({time.time() - t0:.0f}s)", flush=True)
            path.write_text(json.dumps(results, indent=1))
