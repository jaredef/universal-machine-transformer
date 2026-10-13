// The trainer's reference implementation, in plain JavaScript: lag3's network (one layer of softmax attention, two
// heads, read at the last of three positions; a ReLU MLP; an unembedding), its hand-derived backward pass, and Adam.
// The WebGPU trainer (train-gpu.js) computes the same thing; this file is what it is checked against, and the
// fallback when there is no GPU. The factory is self-contained so it can also run inside a Web Worker.
function TrainCoreFactory() {
  "use strict";
  var D = 64, H = 2, DH = 32, HID = 256, SCALE = 1 / Math.sqrt(DH);

  // ---------------------------------------------------------------- parameters, in one flat array
  // Each tensor starts on a 64-float (256-byte) boundary, so the GPU trainer can bind it directly.
  function layout(nin, nout) {
    var spec = [["E", nin, D], ["P", 3, D], ["Wq", D, D], ["Wk", D, D], ["Wv", D, D], ["Wo", D, D],
                ["W1", HID, D], ["b1", 1, HID], ["W2", D, HID], ["b2", 1, D], ["U", nout, D], ["bU", 1, nout]];
    var off = {}, n = 0;
    spec.forEach(function (s) { off[s[0]] = { at: n, rows: s[1], cols: s[2], size: s[1] * s[2] }; n += Math.ceil(s[1] * s[2] / 64) * 64; });
    return { nin: nin, nout: nout, off: off, size: n, names: spec.map(function (s) { return s[0]; }) };
  }

  // A small seeded generator (sfc32), and normal draws by Box-Muller.
  function rng(seed) {
    var a = 0x9e3779b9, b = 0x243f6a88, c = 0xb7e15162, d = seed >>> 0;
    function next() {
      a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
      var t = (a + b) | 0; a = b ^ (b >>> 9); b = (c + (c << 3)) | 0; c = (c << 21) | (c >>> 11); d = (d + 1) | 0;
      t = (t + d) | 0; c = (c + t) | 0;
      return (t >>> 0) / 4294967296;
    }
    for (var i = 0; i < 16; i++) next();
    var spare = null;
    next.normal = function () {
      if (spare !== null) { var s = spare; spare = null; return s; }
      var u = 0; while (u === 0) u = next();
      var v = next(), r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v); return r * Math.cos(2 * Math.PI * v);
    };
    return next;
  }

  // PyTorch's default initialization for these layers: embeddings N(0, 1); P = N(0, 1) * 0.1 (as in lag3's script);
  // each linear layer's weight and bias uniform on +-1/sqrt(fan_in).
  function init(L, seed) {
    var p = new Float32Array(L.size), r = rng(seed), o = L.off;
    function fill(name, f) { for (var i = 0; i < o[name].size; i++) p[o[name].at + i] = f(); }
    fill("E", r.normal); fill("P", function () { return 0.1 * r.normal(); });
    [["Wq", D], ["Wk", D], ["Wv", D], ["Wo", D], ["W1", D], ["b1", D], ["W2", HID], ["b2", HID], ["U", D], ["bU", D]].forEach(function (s) {
      var bound = 1 / Math.sqrt(s[1]); fill(s[0], function () { return (2 * r() - 1) * bound; });
    });
    return p;
  }

  // ---------------------------------------------------------------- forward and backward, one example at a time
  // Scratch space for one example.
  function scratch(L) {
    var f = function (n) { return new Float32Array(n); };
    return { x: f(3 * D), q: f(D), k: f(3 * D), v: f(3 * D), a: f(H * 3), y: f(D), h1: f(D), z: f(HID), r: f(HID), h2: f(D),
             lg: f(L.nout), dh2: f(D), dr: f(HID), dh1: f(D), dy: f(D), dq: f(D), dk: f(3 * D), dv: f(3 * D), dx: f(3 * D) };
  }

  // y[i] = sum_j W[i][j] x[j] (+ b[i]); W is rows x cols at p[w..], x at xs[xo..].
  function mv(p, w, rows, cols, xs, xo, out, oo, b) {
    for (var i = 0; i < rows; i++) {
      var s = b === undefined ? 0 : p[b + i], row = w + i * cols;
      for (var j = 0; j < cols; j++) s += p[row + j] * xs[xo + j];
      out[oo + i] = s;
    }
  }
  // out[j] += sum_i W[i][j] g[i]  (the transpose)
  function mvT(p, w, rows, cols, g, go, out, oo) {
    for (var i = 0; i < rows; i++) {
      var gi = g[go + i], row = w + i * cols;
      if (gi === 0) continue;
      for (var j = 0; j < cols; j++) out[oo + j] += p[row + j] * gi;
    }
  }
  // G[i][j] += g[i] x[j]
  function outer(G, w, rows, cols, g, go, xs, xo) {
    for (var i = 0; i < rows; i++) {
      var gi = g[go + i], row = w + i * cols;
      if (gi === 0) continue;
      for (var j = 0; j < cols; j++) G[row + j] += gi * xs[xo + j];
    }
  }

  function forward(L, p, ids, s) {
    var o = L.off, x = s.x;
    for (var t = 0; t < 3; t++) for (var c = 0; c < D; c++) x[t * D + c] = p[o.E.at + ids[t] * D + c] + p[o.P.at + t * D + c];
    mv(p, o.Wq.at, D, D, x, 2 * D, s.q, 0);
    for (t = 0; t < 3; t++) { mv(p, o.Wk.at, D, D, x, t * D, s.k, t * D); mv(p, o.Wv.at, D, D, x, t * D, s.v, t * D); }
    s.y.fill(0);
    for (var h = 0; h < H; h++) {
      var lo = h * DH, sc = [0, 0, 0];
      for (t = 0; t < 3; t++) { var acc = 0; for (c = lo; c < lo + DH; c++) acc += s.q[c] * s.k[t * D + c]; sc[t] = acc * SCALE; }
      var m = Math.max(sc[0], sc[1], sc[2]), e0 = Math.exp(sc[0] - m), e1 = Math.exp(sc[1] - m), e2 = Math.exp(sc[2] - m), zz = e0 + e1 + e2;
      s.a[h * 3] = e0 / zz; s.a[h * 3 + 1] = e1 / zz; s.a[h * 3 + 2] = e2 / zz;
      for (t = 0; t < 3; t++) { var at = s.a[h * 3 + t]; for (c = lo; c < lo + DH; c++) s.y[c] += at * s.v[t * D + c]; }
    }
    mv(p, o.Wo.at, D, D, s.y, 0, s.h1, 0);
    for (c = 0; c < D; c++) s.h1[c] += x[2 * D + c];
    mv(p, o.W1.at, HID, D, s.h1, 0, s.z, 0, o.b1.at);
    for (var u = 0; u < HID; u++) s.r[u] = s.z[u] > 0 ? s.z[u] : 0;
    mv(p, o.W2.at, D, HID, s.r, 0, s.h2, 0, o.b2.at);
    for (c = 0; c < D; c++) s.h2[c] += s.h1[c];
    mv(p, o.U.at, L.nout, D, s.h2, 0, s.lg, 0, o.bU.at);
  }

  function argmax(a) { var b = 0; for (var i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

  // Cross-entropy for one example, with its gradient (times `scale`) added into G. Returns the example's loss.
  function backward(L, p, G, ids, label, s, scale) {
    var o = L.off, lg = s.lg, nout = L.nout, m = lg[0];
    for (var i = 1; i < nout; i++) if (lg[i] > m) m = lg[i];
    var zz = 0;
    for (i = 0; i < nout; i++) { lg[i] = Math.exp(lg[i] - m); zz += lg[i]; }
    var loss = -Math.log(lg[label] / zz);
    for (i = 0; i < nout; i++) lg[i] = lg[i] / zz * scale;          // lg is now dlogits
    lg[label] -= scale;
    outer(G, o.U.at, nout, D, lg, 0, s.h2, 0);
    for (i = 0; i < nout; i++) G[o.bU.at + i] += lg[i];
    s.dh2.fill(0); mvT(p, o.U.at, nout, D, lg, 0, s.dh2, 0);
    outer(G, o.W2.at, D, HID, s.dh2, 0, s.r, 0);
    for (i = 0; i < D; i++) G[o.b2.at + i] += s.dh2[i];
    s.dr.fill(0); mvT(p, o.W2.at, D, HID, s.dh2, 0, s.dr, 0);
    for (i = 0; i < HID; i++) if (s.z[i] <= 0) s.dr[i] = 0;          // dr is now dz
    outer(G, o.W1.at, HID, D, s.dr, 0, s.h1, 0);
    for (i = 0; i < HID; i++) G[o.b1.at + i] += s.dr[i];
    for (i = 0; i < D; i++) s.dh1[i] = s.dh2[i];
    mvT(p, o.W1.at, HID, D, s.dr, 0, s.dh1, 0);
    outer(G, o.Wo.at, D, D, s.dh1, 0, s.y, 0);
    s.dy.fill(0); mvT(p, o.Wo.at, D, D, s.dh1, 0, s.dy, 0);
    s.dq.fill(0); s.dk.fill(0);
    for (var h = 0; h < H; h++) {
      var lo = h * DH, da = [0, 0, 0], t, c;
      for (t = 0; t < 3; t++) {
        var at = s.a[h * 3 + t], acc = 0;
        for (c = lo; c < lo + DH; c++) { s.dv[t * D + c] = at * s.dy[c]; acc += s.dy[c] * s.v[t * D + c]; }
        da[t] = acc;
      }
      var mean = s.a[h * 3] * da[0] + s.a[h * 3 + 1] * da[1] + s.a[h * 3 + 2] * da[2];
      for (t = 0; t < 3; t++) {
        var ds = s.a[h * 3 + t] * (da[t] - mean) * SCALE;
        for (c = lo; c < lo + DH; c++) { s.dq[c] += ds * s.k[t * D + c]; s.dk[t * D + c] = ds * s.q[c]; }
      }
    }
    outer(G, o.Wq.at, D, D, s.dq, 0, s.x, 2 * D);
    s.dx.fill(0);
    for (var t2 = 0; t2 < 3; t2++) {
      outer(G, o.Wk.at, D, D, s.dk, t2 * D, s.x, t2 * D);
      outer(G, o.Wv.at, D, D, s.dv, t2 * D, s.x, t2 * D);
      mvT(p, o.Wk.at, D, D, s.dk, t2 * D, s.dx, t2 * D);
      mvT(p, o.Wv.at, D, D, s.dv, t2 * D, s.dx, t2 * D);
    }
    mvT(p, o.Wq.at, D, D, s.dq, 0, s.dx, 2 * D);
    for (i = 0; i < D; i++) s.dx[2 * D + i] += s.dh1[i];
    for (t2 = 0; t2 < 3; t2++) for (i = 0; i < D; i++) {
      G[o.E.at + ids[t2] * D + i] += s.dx[t2 * D + i];
      G[o.P.at + t2 * D + i] += s.dx[t2 * D + i];
    }
    return loss;
  }

  // The summed loss and gradient over examples `which` (indices into the table), each scaled by 1/n so that the
  // gradients of several chunks add up to the full-batch mean. Returns {loss (sum), correct}.
  function grad(L, p, G, table, which, n, s) {
    s = s || scratch(L);
    var ids = [0, 0, 0], loss = 0, correct = 0;
    for (var e = 0; e < which.length; e++) {
      var k = which[e];
      ids[0] = table.steps[3 * k]; ids[1] = table.steps[3 * k + 1]; ids[2] = table.steps[3 * k + 2];
      forward(L, p, ids, s);
      if (argmax(s.lg) === table.labels[k]) correct++;
      loss += backward(L, p, G, ids, table.labels[k], s, 1 / n);
    }
    return { loss: loss, correct: correct };
  }

  // Predictions for examples `which` (all of them when omitted), as an Int32Array of output indices.
  function predict(L, p, table, which) {
    var s = scratch(L), n = which ? which.length : table.labels.length, out = new Int32Array(n), ids = [0, 0, 0];
    for (var e = 0; e < n; e++) {
      var k = which ? which[e] : e;
      ids[0] = table.steps[3 * k]; ids[1] = table.steps[3 * k + 1]; ids[2] = table.steps[3 * k + 2];
      forward(L, p, ids, s); out[e] = argmax(s.lg);
    }
    return out;
  }

  // PyTorch's Adam (betas 0.9, 0.999; eps 1e-8), on the flat arrays.
  function adam(p, G, M, V, step, lr) {
    var b1 = 0.9, b2 = 0.999, eps = 1e-8, c1 = 1 - Math.pow(b1, step), c2 = Math.sqrt(1 - Math.pow(b2, step));
    for (var i = 0; i < p.length; i++) {
      var g = G[i];
      M[i] = b1 * M[i] + (1 - b1) * g; V[i] = b2 * V[i] + (1 - b2) * g * g;
      p[i] -= (lr / c1) * M[i] / (Math.sqrt(V[i]) / c2 + eps);
    }
  }

  return { D: D, H: H, DH: DH, HID: HID, layout: layout, rng: rng, init: init, scratch: scratch, forward: forward,
           backward: backward, grad: grad, predict: predict, adam: adam, argmax: argmax };
}
(function (root) {
  var api = TrainCoreFactory();
  api.factory = TrainCoreFactory;
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.TrainCore = api;
})(this);
