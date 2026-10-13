"""Compare web/train-core.js with PyTorch (float64): the loss and every gradient on one batch, and 30 steps of Adam.
Usage: node gradcheck.js out.json && python gradcheck.py out.json   (needs torch)"""
import json, math, sys, torch
J = json.load(open(sys.argv[1])); torch.set_default_dtype(torch.float64)
off, nin, nout = J["off"], J["nin"], J["nout"]; P = torch.tensor(J["params"])
def t(name): o = off[name]; return P[o["at"]:o["at"] + o["size"]].view(o["rows"], o["cols"]).clone().requires_grad_(True)
W = {n: t(n) for n in off}
steps = torch.tensor(J["steps"]).view(-1, 3); labels = torch.tensor(J["labels"])
D = J.get("d", 64); DH = D // 2
def model(W, idx):
    x = W["E"][idx] + W["P"]; n = x.shape[0]
    q = (x[:, 2] @ W["Wq"].T).view(n, 2, DH); k = (x @ W["Wk"].T).view(n, 3, 2, DH); v = (x @ W["Wv"].T).view(n, 3, 2, DH)
    a = torch.softmax(torch.einsum("nhc,nthc->nht", q, k) / math.sqrt(DH), -1)
    y = torch.einsum("nht,nthc->nhc", a, v).reshape(n, D)
    h = x[:, 2] + y @ W["Wo"].T
    h = h + torch.relu(h @ W["W1"].T + W["b1"][0]) @ W["W2"].T + W["b2"][0]
    return h @ W["U"].T + W["bU"][0]
b = torch.tensor(J["batch"]); loss = torch.nn.functional.cross_entropy(model(W, steps[b]), labels[b]); loss.backward()
print(f"loss torch {loss.item():.7f}  js {J['loss']:.7f}")
G = torch.tensor(J["grads"]); worst = 0
for n, w in W.items():
    o = off[n]; g = G[o["at"]:o["at"] + o["size"]].view(o["rows"], o["cols"])
    rel = ((g - w.grad).abs().max() / (w.grad.abs().max() + 1e-12)).item(); worst = max(worst, rel)
    print(f"  {n:3} max|grad| {w.grad.abs().max().item():.3e}  max rel diff {rel:.2e}")
print("worst relative gradient difference", f"{worst:.2e}")
W2 = {n: t(n) for n in off}; opt = torch.optim.Adam(W2.values(), lr=3e-3); tr = torch.tensor(J["tr"]); curve = []
for s in range(30):
    l = torch.nn.functional.cross_entropy(model(W2, steps[tr]), labels[tr]); curve.append(l.item()); opt.zero_grad(); l.backward(); opt.step()
print("adam curve max abs diff", max(abs(a - b) for a, b in zip(curve, J["curve"])), f"first {curve[0]:.5f} last {curve[-1]:.5f} js last {J['curve'][-1]:.5f}")
