// Checks what inside.html shows: the values Network.forward returns along the way are the pass (each step recomputed from
// the one before, ending at the same logits), on every window of the first 200,000 passes of the sieve to 120 and on the
// three windows it never saw (inside-data.js), which the network must get right.  node inside-check.js
global.window = global;
require("../data2.js"); require("../inside-data.js");
const TCT = require("../core.js"), P = require("../primes.js");
const D = window.TCT_DATA2, W = D.weights, e = new TCT.Engine(D), net = e.net;
const lin = (M, x, b) => M.map((r, i) => r.reduce((s, w, j) => s + w * x[j], b ? b[i] : 0));
let worst = 0, n = 0;
function close(a, b, what) {
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]) / Math.max(1, Math.abs(b[i]));
    worst = Math.max(worst, d);
    if (d > 1e-9) throw new Error(what + "[" + i + "]: " + a[i] + " vs " + b[i]);
  }
}
function check(ids) {
  const f = net.forward(ids);
  ids.forEach((id, p) => close(f.x[p], W["E.weight"][id].map((v, c) => v + W.P[p][c]), "x"));
  close(f.q, lin(W["Wq.weight"], f.x[2]), "q");
  f.x.forEach((x, p) => { close(f.k[p], lin(W["Wk.weight"], x), "k"); close(f.v[p], lin(W["Wv.weight"], x), "v"); });
  const y = new Array(8).fill(0);
  [0, 1].forEach((h) => {
    const a = f.attention[h], sum = a[0] + a[1] + a[2];
    if (Math.abs(sum - 1) > 1e-12) throw new Error("attention does not sum to 1");
    for (let p = 0; p < 3; p++) for (let c = 4 * h; c < 4 * h + 4; c++) y[c] += a[p] * f.v[p][c];
  });
  close(f.y, y, "y"); close(f.o, lin(W["Wo.weight"], f.y), "o");
  close(f.h1, f.x[2].map((v, c) => v + f.o[c]), "h1");
  close(f.pre, lin(W["W1.weight"], f.h1, W["W1.bias"][0]), "pre"); close(f.hidden, Array.from(f.pre, (v) => Math.max(0, v)), "hidden");
  close(f.mlp, lin(W["W2.weight"], f.hidden, W["W2.bias"][0]), "mlp"); close(f.h2, f.h1.map((v, c) => v + f.mlp[c]), "h2");
  close(f.logits, lin(W["U.weight"], f.h2, W["U.bias"][0]), "logits");
  n++;
  return f;
}
const s = P.sieve(120), run = new TCT.Run(e, TCT.parseMachine(s.start, s.rules), s.tape), seen = new Set();
while (run.passes < 200000) {
  const ids = run.window(), key = e.key(ids);
  if (!seen.has(key)) { seen.add(key); check(ids); }
  run.step(false);
}
window.TCT_INSIDE.unseen.forEach((u) => {
  const ids = u.window.map((t) => e.id[t]), f = check(ids), wrote = D.emits[D.vout[f.best]].join(" ");
  if (e.trained.has(e.key(ids))) throw new Error(u.window.join(" ") + " is in training");
  if (wrote !== u.wrote || wrote !== e.exact(ids)) throw new Error(u.window.join(" ") + ": wrote " + wrote + ", exact " + e.exact(ids));
});
console.log("inside: " + n + " windows, every value forward returns recomputed from the one before (largest relative difference " +
  worst.toExponential(1) + "); the 3 windows never in training are right");
