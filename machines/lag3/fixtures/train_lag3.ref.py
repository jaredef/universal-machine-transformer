#!/usr/bin/env python3
"""Train ONE transformer on the steps of a universal Turing machine (reference script; needs torch).

Training data: the lagging-window steps of the universal machine (derived/utm.py) while it runs RANDOM small machines
(1-5 states, half of them biased to travel right) on random tapes of 1-6 cells over 0 1 _. No meaningful program is in
the training data. The witness (derived/lag3w.py) then runs held-out machines (increment, decrement, invert, parity,
divisibility by 3, unary addition, a six-state palindrome checker) on longer inputs with the trained step.

Writes lag3_training.json: configuration, training curves per seed, the committed weights (seed 0), the training
windows, and the trained model's predictions on them.
"""
import json, math, random, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "derived"))
import lag3, lag1, lag0  # noqa: E402

N_MACHINES, SEEDS, STEPS = 4000, 3, 3000
t0 = time.time()

# ---- the universal machine's token vocabularies (identical for every described machine)
REF = lag3.utm_program(lag3.held_out_source("invert", "0"))
VIN = lag0.alphabet(REF)
NET = lag1.Net(REF)
VOUT = [name for name, _, _ in NET.vocab_out]
EMIT2NAME = {tuple(e): n for n, e, _ in NET.vocab_out}

# ---- training windows: the universal machine running random machines on random tapes
rng = random.Random(2)
windows = set()
for i in range(N_MACHINES):
    src = lag3.random_source(rng, rng.randint(1, 5), "".join(rng.choice("01_") for _ in range(rng.randint(1, 6))),
                             "RRRLS" if i % 2 else "LRS", 0.5)
    p = lag3.utm_program(src)
    lag3.lag_run(p, lag3.Exact(p), max_decode=400_000, seen=windows)
W = sorted(windows)
Y = [EMIT2NAME[tuple(lag0.rule(REF, *w) or [])] for w in W]
print(f"{len(W)} training windows from {N_MACHINES} random machines ({time.time() - t0:.0f}s)", flush=True)


class Model(torch.nn.Module):
    def __init__(s, d=64, heads=2, hidden=256):
        super().__init__()
        s.d, s.h = d, heads
        s.E = torch.nn.Embedding(len(VIN), d); s.P = torch.nn.Parameter(torch.randn(3, d) * 0.1)
        s.Wq, s.Wk, s.Wv, s.Wo = (torch.nn.Linear(d, d, bias=False) for _ in range(4))
        s.W1, s.W2 = torch.nn.Linear(d, hidden), torch.nn.Linear(hidden, d)
        s.U = torch.nn.Linear(d, len(VOUT))
    def forward(s, idx, return_attn=False):
        x = s.E(idx) + s.P
        n, t, d = x.shape; dh = d // s.h
        q = s.Wq(x[:, -1:]).view(n, 1, s.h, dh).transpose(1, 2)
        k = s.Wk(x).view(n, t, s.h, dh).transpose(1, 2)
        v = s.Wv(x).view(n, t, s.h, dh).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
        y = (a @ v).transpose(1, 2).reshape(n, 1, d)
        h = x[:, -1:] + s.Wo(y)
        h = h + s.W2(torch.relu(s.W1(h)))
        logits = s.U(h[:, 0])
        return (logits, a[:, :, 0]) if return_attn else logits


X = torch.tensor([[VIN.index(t) for t in w] for w in W])
T = torch.tensor([VOUT.index(y) for y in Y])
out = {"architecture": {"d_model": 64, "heads": 2, "mlp_hidden": 256, "attention": "softmax, read at the last position",
                        "optimizer": "Adam lr=3e-3 full batch", "steps": STEPS},
       "training_data": {"random_machines": N_MACHINES, "states": "1-5", "tapes": "1-6 cells over 0 1 _",
                         "moves": "half uniform L R S, half biased right (R R R L S)", "windows": len(W)},
       "vin": VIN, "vout": VOUT, "runs": []}
for seed in range(SEEDS):
    torch.manual_seed(seed)
    m = Model(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    curve, fit_at = [], None
    for step in range(STEPS + 1):
        logits = m(X)
        loss = torch.nn.functional.cross_entropy(logits, T)
        if step % 100 == 0:
            acc = (logits.argmax(-1) == T).float().mean().item()
            curve.append({"step": step, "train_acc": round(acc, 5), "loss": round(loss.item(), 5)})
            if acc == 1.0 and fit_at is None:
                fit_at = step
        if step < STEPS:
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits, attn = m(X, return_attn=True)
        top2 = logits.topk(2, dim=-1).values
        pred = logits.argmax(-1)
        right = (pred == T)
        margin = (top2[:, 0] - top2[:, 1])[right].min().item() if right.any() else 0.0
    rec = {"seed": seed, "fit_at": fit_at, "final_train_acc": curve[-1]["train_acc"],
           "train_errors": int((~right).sum()), "min_margin_on_correct": round(margin, 4),
           "attention_by_head": [[round(x, 3) for x in h] for h in attn.mean(0).tolist()], "curve": curve}
    out["runs"].append(rec)
    print({k: v for k, v in rec.items() if k != "curve"}, f"({time.time() - t0:.0f}s)", flush=True)
    if seed == 0:
        sd = m.state_dict()
        out["weights"] = {k: [[round(float(x), 6) for x in row] for row in (v.unsqueeze(0) if v.dim() == 1 else v).tolist()]
                          for k, v in sd.items()}
        out["train_windows"] = ["\u0001".join(w) for w in W]
        out["train_predictions"] = [VOUT[i] for i in pred.tolist()]
(HERE / "lag3_training.json").write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote lag3_training.json ({(HERE / 'lag3_training.json').stat().st_size // 1024} KB, {time.time() - t0:.0f}s)")
