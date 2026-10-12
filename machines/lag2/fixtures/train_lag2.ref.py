#!/usr/bin/env python3
"""Train an ordinary small transformer to be the step of the lagging-window decoder (reference script; needs torch).

The model knows nothing of the derived construction in lag1: learned token and position embeddings, one layer of
softmax attention (2 heads), a ReLU MLP, an unembedding, read at the last position of the 3-token window. It is trained
by full-batch Adam on cross-entropy against lag0's rule table, but ONLY on the windows the decoder meets while running
binary increment on short inputs. It is then tested as the decoder's step on much longer inputs.

    main     train on the windows of every input of 1..3 bits  (they include every window longer inputs ever meet)
    control  train on the windows of every input of 1..2 bits  (some windows needed by longer inputs are missing)

Writes lag2_training.json: the configuration, per-seed curves, the committed weights of the main run (seed 0),
and the model's prediction on every reachable window, for the torch-free witness (derived/lag2.py) to check.
"""
import itertools, json, math, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lag0" / "derived")); sys.path.insert(0, str(ROOT / "lag1" / "derived"))
sys.path.insert(0, str(ROOT / "tm0"))
import lag0, lag1, tm0  # noqa: E402

SRC = (ROOT / "lag0" / "fixtures" / "binary-increment.tm0").read_text()
def prog_for(bits): return lag0.parse_tm0(SRC.replace("tape 1011", "tape " + bits))
def inputs(lo, hi): return ["".join(b) for n in range(lo, hi + 1) for b in itertools.product("01", repeat=n)]

BASE = prog_for("1011")
REF = lag1.Net(BASE)                      # only for the token vocabularies: vocab_in and vocab_out
VIN = REF.vocab_in
VOUT = [name for name, _, _ in REF.vocab_out]
EMITS = {name: emits for name, emits, _ in REF.vocab_out}
REACH = sorted(lag1.reachable_windows(BASE))

def target(w):
    out = lag0.rule(BASE, *w)
    for name, emits, _ in REF.vocab_out:
        if emits == (out or []):
            return name
    raise KeyError(w)

def met_windows(bits):
    p = prog_for(bits); table = lag0.compile_rules(p)
    ring = lag0.initial_ring(p); T = [ring[-1], *ring]; k = 0; seen = set()
    while True:
        w = tuple(T[k:k + 3]); seen.add(w); out = table.get(w)
        if out is None: return seen
        T.extend(out); k += 1

class Model(torch.nn.Module):
    def __init__(s, d=32, heads=2, hidden=64):
        super().__init__()
        s.d, s.h = d, heads
        s.E = torch.nn.Embedding(len(VIN), d); s.P = torch.nn.Parameter(torch.randn(3, d) * 0.1)
        s.Wq, s.Wk, s.Wv, s.Wo = (torch.nn.Linear(d, d, bias=False) for _ in range(4))
        s.W1, s.W2 = torch.nn.Linear(d, hidden), torch.nn.Linear(hidden, d)
        s.U = torch.nn.Linear(d, len(VOUT))
    def forward(s, idx, return_attn=False):
        x = s.E(idx) + s.P                                          # (n, 3, d)
        n, t, d = x.shape; dh = d // s.h
        q = s.Wq(x[:, -1:]).view(n, 1, s.h, dh).transpose(1, 2)     # read at the last position
        k = s.Wk(x).view(n, t, s.h, dh).transpose(1, 2)
        v = s.Wv(x).view(n, t, s.h, dh).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)   # (n, h, 1, 3)
        y = (a @ v).transpose(1, 2).reshape(n, 1, d)
        h = x[:, -1:] + s.Wo(y)
        h = h + s.W2(torch.relu(s.W1(h)))
        logits = s.U(h[:, 0])
        return (logits, a[:, :, 0]) if return_attn else logits

def encode(ws): return torch.tensor([[VIN.index(t) for t in w] for w in ws])

class TableNet:                                                    # lets lag1.decode use a trained model
    def __init__(s, model):
        with torch.no_grad():
            pred = model(encode(REACH)).argmax(-1).tolist()
        s.out = {w: VOUT[i] for w, i in zip(REACH, pred)}
    def forward(s, window):
        name = s.out[tuple(window)]; return name, EMITS[name]

def run_ok(net, bits):
    p = prog_for(bits)
    want = tm0.run(tm0.parse_program(SRC.replace("tape 1011", "tape " + bits)), 10_000, False)
    v, steps, tape_d = want
    lo, hi = min(min(tape_d), 0), max(max(tape_d), 0)
    want_line = f"{v} steps={steps} tape={''.join(tape_d.get(i, '_') for i in range(lo, hi + 1))}"
    try:
        verdict, s, _, tape = lag1.decode(p, net, "lag", max_steps=2_000)
    except (AssertionError, KeyError, ValueError):               # an untrained step can wreck the transcript
        return False
    return f"{verdict} steps={s} tape={tape}" == want_line

def train(train_bits, seed, steps=3000, every=50, test_bits=None):
    torch.manual_seed(seed)
    W = sorted(set().union(*(met_windows(b) for b in train_bits)))
    X, Y = encode(W), torch.tensor([VOUT.index(target(w)) for w in W])
    m = Model(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    curve, fit_at, runs_at = [], None, None
    for step in range(steps + 1):
        if step % every == 0:
            with torch.no_grad():
                acc = (m(X).argmax(-1) == Y).float().mean().item()
            net = TableNet(m)
            run_acc = sum(run_ok(net, b) for b in test_bits) / len(test_bits)
            curve.append({"step": step, "train_acc": round(acc, 4), "long_run_acc": round(run_acc, 4),
                          "loss": round(torch.nn.functional.cross_entropy(m(X), Y).item(), 5)})
            if acc == 1.0 and fit_at is None: fit_at = step
            if run_acc == 1.0 and runs_at is None: runs_at = step
        if step < steps:
            opt.zero_grad(); torch.nn.functional.cross_entropy(m(X), Y).backward(); opt.step()
    return m, W, curve, fit_at, runs_at

def weights(m):
    return {k: [[round(float(x), 7) for x in row] for row in (v.unsqueeze(0) if v.dim() == 1 else v).tolist()]
            if v.dim() <= 2 else None for k, v in m.state_dict().items()}

def main():
    test_bits = inputs(4, 8)                                      # 496 longer inputs, checked during training
    out = {"program": "binary-increment", "architecture": {"d_model": 32, "heads": 2, "mlp_hidden": 64,
           "attention": "softmax, read at the last position", "optimizer": "Adam lr=3e-3 full batch", "steps": 3000},
           "vin": VIN, "vout": VOUT, "runs": {}}
    for regime, bits in (("main", inputs(1, 3)), ("control", inputs(1, 2))):
        out["runs"][regime] = []
        for seed in range(5):
            m, W, curve, fit_at, runs_at = train(bits, seed, test_bits=test_bits)
            net = TableNet(m)
            with torch.no_grad():
                logits, attn = m(encode(W), return_attn=True)
                top2 = logits.topk(2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1]).min().item()
                attn_mean = attn.mean(0).tolist()                 # per head: mean attention to positions 0,1,2
            rec = {"seed": seed, "train_windows": len(W), "fit_at": fit_at, "long_runs_all_correct_at": runs_at,
                   "final_train_acc": curve[-1]["train_acc"], "final_long_run_acc": curve[-1]["long_run_acc"],
                   "min_train_margin": round(margin, 4), "attention_by_head": [[round(x, 3) for x in h] for h in attn_mean],
                   "curve": curve}
            if regime == "control":
                needed = set().union(*(met_windows(b) for b in test_bits))
                missing = sorted(needed - set(W))
                rec["missing_windows"] = len(missing)
                rec["missing_predicted_correctly"] = sum(net.out[w] == target(w) for w in missing)
            out["runs"][regime].append(rec)
            if regime == "main" and seed == 0:
                out["weights"] = weights(m)
                out["predictions"] = {"|".join(w): net.out[w] for w in REACH}
                out["train_windows"] = ["|".join(w) for w in W]
            if regime == "control" and seed == 0:            # kept so the curriculum can run the control live
                out["control_weights"] = weights(m)
                out["control_predictions"] = {"|".join(w): net.out[w] for w in REACH}
                out["control_train_windows"] = ["|".join(w) for w in W]
            print(regime, seed, {k: rec[k] for k in rec if k not in ("curve",)}, flush=True)
    (HERE / "lag2_training.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
