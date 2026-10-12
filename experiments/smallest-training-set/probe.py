#!/usr/bin/env python3
"""Retrain one network (default: 2,250 machines, seed 3) and show what it predicts on chosen steps (needs torch).

    python probe.py [N SEED]

At 2,250 machines every seed tried misses one step a demo needs, `1 0 go_aS:I` (the universal machine writes 0 and
stays on a marked 1; the head steps back, so the answer is `ret:` plus the middle token). The probe shows the network's
answer and the neighbouring steps it did or did not see in training.
"""
import sys
import torch
args = sys.argv[1:3]
sys.argv = sys.argv[:1]
import train as T  # noqa: E402

n, seed = (int(args[0]), int(args[1])) if len(args) == 2 else (2250, 3)
tr = ((T.FIRST >= 0) & (T.FIRST < n)).nonzero().squeeze(1)
X, Y = T.X_all[tr], T.Y_all[tr]
torch.manual_seed(seed)
m = T.Model().to(T.DEV); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
for _ in range(T.STEPS):
    loss = torch.nn.functional.cross_entropy(m(X), Y); opt.zero_grad(); loss.backward(); opt.step()
S = T.D["steps"]
print(f"N={n} seed={seed}")
with torch.no_grad():
    for w in ["1 0 go_aS:I", "_ 0 go_aS:I", "1 1 go_aS:I", "0 0 go_aS:I", "1 0 go_bS:I", "1 0 go_aS:O"]:
        i = S.index(w.replace(" ", "\u0001"))
        p = torch.softmax(m(T.X_all[i:i + 1]), -1)[0]
        top = p.topk(2)
        seen = 0 <= T.D["first"][i] < n
        print(f"  {w:12} correct {T.D['labels'][i]:6} {'in training ' if seen else 'never seen  '} -> "
              + ", ".join(f"{T.VOUT[j]} {v:.2f}" for v, j in zip(top.values.tolist(), top.indices.tolist())))
