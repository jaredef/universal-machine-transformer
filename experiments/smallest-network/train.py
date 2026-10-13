#!/usr/bin/env python3
"""How small can the trained universal network be? (needs torch)

Trains lag3's architecture (one layer of softmax attention, 2 heads, read at the last of 3 positions; a ReLU MLP; an
unembedding) at a range of widths, on the universal machine's whole step table: all 14,849 steps that 16,000 random
machines produce (../smallest-training-set/steps.json), with the vocabularies trimmed to the tokens the table uses (429
in, 434 out). A network passes when it gets every step of the table exactly right: then it runs every program whose
steps lie in the table, which every program tried so far does.

    python train.py                     every size in GRID, seeds 0-2
    python train.py 16:64 12:48         chosen sizes (width:mlp)
    SEEDS=3,4 LR=0.01 STEPS=20000 python train.py 12:48

Writes results.json (merged with earlier results) and, for every network that fits, its weights in fixtures/.
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
TESTS = {k: torch.tensor(v, device=DEV) for k, v in S["tests"].items()}

GRID = [(64, 256), (48, 192), (32, 128), (24, 96), (16, 64), (12, 48), (8, 32)]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
LR, MAX_STEPS = float(os.environ.get("LR", "0.003")), int(os.environ.get("STEPS", "8000"))


class Model(torch.nn.Module):                            # lag3's architecture, at width d and MLP width hidden
    def __init__(s, d, hidden, heads=2):
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


def train(d, hidden, seed):
    torch.manual_seed(seed)
    m = Model(d, hidden).to(DEV)
    params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    fit_at, best_err, t0 = None, None, time.time()
    for step in range(1, MAX_STEPS + 1):
        logits = m(X)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                err = int((m(X).argmax(-1) != Y).sum())
            best_err = err if best_err is None else min(best_err, err)
            if err == 0:
                fit_at = step
                break
    with torch.no_grad():
        right = m(X).argmax(-1) == Y
    rec = {"d": d, "hidden": hidden, "seed": seed, "lr": LR, "max_steps": MAX_STEPS, "params": params,
           "fit_at": fit_at, "errors": int((~right).sum()), "steps_run": step, "seconds": round(time.time() - t0, 1),
           "programs_correct": sum(bool(right[idx].all()) for idx in TESTS.values()), "device": DEV.type}
    if rec["errors"] == 0:
        (HERE / "fixtures").mkdir(exist_ok=True)
        w = {k: [[round(float(x), 6) for x in row] for row in (v.unsqueeze(0) if v.dim() == 1 else v).tolist()]
             for k, v in m.state_dict().items()}
        (HERE / "fixtures" / f"net_d{d}_h{hidden}_seed{seed}.json").write_text(json.dumps(
            {"d": d, "hidden": hidden, "heads": 2, "seed": seed, "lr": LR, "params": params, "vin": VIN, "vout": VOUT,
             "weights": w}, separators=(",", ":")))
    return rec


if __name__ == "__main__":
    grid = [tuple(int(x) for x in a.split(":")) for a in sys.argv[1:]] or GRID
    path = HERE / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {"runs": []}
    for d, hidden in grid:
        for seed in SEEDS:
            r = train(d, hidden, seed)
            results["runs"] = [x for x in results["runs"]
                               if (x["d"], x["hidden"], x["seed"], x["lr"], x["max_steps"]) != (d, hidden, seed, LR, MAX_STEPS)] + [r]
            path.write_text(json.dumps(results, indent=1))
            print(f"d={d:>3} mlp={hidden:>4} seed={seed} params={r['params']:>7,} lr={LR} -> "
                  f"{'FITS at step ' + str(r['fit_at']) if r['fit_at'] else str(r['errors']) + ' steps wrong'}; "
                  f"programs {r['programs_correct']}/12; {r['seconds']}s", flush=True)
