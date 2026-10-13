const W = __dirname + "/../";  // checks web/train-core.js against PyTorch: node gradcheck.js out.json && python gradcheck.py out.json
global.window = {}; require(W + "table.js"); const T = window.TCT_TABLE; const C = require(W + "train-core.js");
const L = C.layout(T.vin.length, T.vout.length), p = C.init(L, 1), p0 = Float32Array.from(p);
const batch = Array.from({ length: 300 }, (_, i) => (i * 49) % T.labels.length);
const G = new Float32Array(L.size), r = C.grad(L, p, G, T, batch, batch.length);
// 30 steps of full-batch Adam on 2,000 steps
const tr = Array.from({ length: 2000 }, (_, i) => (i * 7) % T.labels.length), M = new Float32Array(L.size), V = new Float32Array(L.size), curve = [];
const q = Float32Array.from(p0);
for (let s = 1; s <= 30; s++) { const g = new Float32Array(L.size); const o = C.grad(L, q, g, T, tr, tr.length); curve.push(o.loss / tr.length); C.adam(q, g, M, V, s, 3e-3); }
require("fs").writeFileSync(process.argv[2], JSON.stringify({ off: L.off, nin: L.nin, nout: L.nout, params: Array.from(p0), grads: Array.from(G),
  loss: r.loss / batch.length, batch, tr, curve, steps: T.steps.slice(0, 3 * T.labels.length), labels: T.labels }));
console.log("js loss", r.loss / batch.length, "curve", curve[0].toFixed(5), curve[29].toFixed(5));
