#!/usr/bin/env python3
"""Milestone 3: train networks on universal machine version 2's step table (needs torch).

The same architecture and recipe as the smallest-network experiment (one layer of softmax attention, 2 heads, read at the
last of 3 positions; a ReLU MLP of 4 x width; an unembedding; full-batch Adam at learning rate 0.003, stopping when every
training step is right), trained on every step the 4,000 random machines of collect2.py produced. Then:

- the tail: the steps the demonstration programs need that no random machine produced (all in the programs spread among 64
  and 256 states): does the network get them right anyway?
- the programs: a program runs correctly exactly when every step it needs is right (the decoder is deterministic).

    python train2.py 64 32 16 8            widths; SEEDS=0,1 for more seeds; writes results2.json, fixtures2/
"""
import json, math, os, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
DATA = os.environ.get("DATA", "steps2.json")                 # steps3.json: with the tour machines (collect_tail.py)
S = json.loads((HERE / DATA).read_text())
VIN, VOUT = S["vin"], S["vout"]
iin, iout = {t: i for i, t in enumerate(VIN)}, {t: i for i, t in enumerate(VOUT)}
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
X = torch.tensor([[iin[t] for t in s.split("\u0001")] for s in S["steps"]], device=DEV)
Y = torch.tensor([iout[l] for l in S["labels"]], device=DEV)
FIRST = torch.tensor(S["first"], device=DEV)
TRAIN = (FIRST >= 0).nonzero().squeeze(1)
TAIL = (FIRST < 0).nonzero().squeeze(1)
TESTS = {k: torch.tensor(v, device=DEV) for k, v in S["tests"].items()}
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
LR, MAX_STEPS = 0.003, int(os.environ.get("STEPS", "20000"))


class Model(torch.nn.Module):
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


def right(m, chunk=40000):
    with torch.no_grad():
        return torch.cat([m(X[i:i + chunk]).argmax(-1) == Y[i:i + chunk] for i in range(0, len(X), chunk)])


def run(d, seed):
    torch.manual_seed(seed)
    m = Model(d).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=LR)
    params = sum(p.numel() for p in m.parameters())
    fit_at, t0 = None, time.time()
    for step in range(1, MAX_STEPS + 1):
        loss = torch.nn.functional.cross_entropy(m(X[TRAIN]), Y[TRAIN])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0:
            ok = right(m)
            wrong = int((~ok[TRAIN]).sum())
            print(f"    width {d} seed {seed} step {step}: loss {loss.item():.4f}, training steps wrong {wrong:,} ({time.time() - t0:.0f}s)", flush=True)
            if wrong == 0:
                fit_at = step; break
    ok = right(m)
    rec = {"d": d, "seed": seed, "params": params, "fit_at": fit_at, "train_steps": len(TRAIN), "train_wrong": int((~ok[TRAIN]).sum()),
           "tail": len(TAIL), "tail_right": int(ok[TAIL].sum()), "seconds": round(time.time() - t0),
           "programs": {k: {"needs": len(v), "wrong": int((~ok[v]).sum())} for k, v in TESTS.items()}}
    if rec["train_wrong"] == 0:
        (HERE / "fixtures2").mkdir(exist_ok=True)
        w = {k: [[round(float(x), 6) for x in row] for row in (v.unsqueeze(0) if v.dim() == 1 else v).tolist()] for k, v in m.state_dict().items()}
        tag = "net2" if DATA == "steps2.json" else "net3"
        (HERE / "fixtures2" / f"{tag}_d{d}_seed{seed}.json").write_text(json.dumps(
            {"d": d, "heads": 2, "seed": seed, "params": params, "vin": VIN, "vout": VOUT, "weights": w}, separators=(",", ":")))
    return rec


if __name__ == "__main__":
    path = HERE / ("results2.json" if DATA == "steps2.json" else "results3.json")
    results = json.loads(path.read_text()) if path.exists() else {"runs": []}
    for d in [int(a) for a in sys.argv[1:]] or [64, 32, 16, 8]:
        for seed in SEEDS:
            r = run(d, seed)
            results["runs"] = [x for x in results["runs"] if (x["d"], x["seed"]) != (d, seed)] + [r]
            path.write_text(json.dumps(results, indent=1))
            bad = [k for k, v in r["programs"].items() if v["wrong"]]
            print(f"width {d} ({r['params']:,} params) seed {seed}: {'learned all ' + format(r['train_steps'], ',') + ' steps at Adam step ' + str(r['fit_at']) if r['fit_at'] else format(r['train_wrong'], ',') + ' training steps wrong'}; "
                  f"tail {r['tail_right']}/{r['tail']} right; programs correct {len(r['programs']) - len(bad)}/{len(r['programs'])}"
                  + (f" (wrong: {', '.join(bad)})" if bad else "") + f" ({r['seconds']}s)", flush=True)
