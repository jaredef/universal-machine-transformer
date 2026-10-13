// The trainer on the GPU (WebGPU). Same network, same hand-derived backward pass, same Adam as train-core.js, over
// the whole training set at once (full batch). Every linear layer and its gradients is one tiled matrix multiply;
// attention, softmax cross-entropy, the embedding gradient, and Adam are small kernels of their own. One training
// step is one command buffer of ~40 dispatches.
(function (root) {
  "use strict";
  var D = 64, HID = 256, SPLIT = 1024;

  // ---------------------------------------------------------------- kernels (WGSL)
  // Every kernel takes the same uniform block: 16 words, read by name below.
  var U = "struct U { a:u32, b:u32, c:u32, d:u32, e:u32, f:u32, g:u32, h:u32, i:u32, j:u32, k:u32, l:u32, m:u32, n:u32, o:u32, p:u32 };\n" +
          "@group(0) @binding(0) var<uniform> u: U;\n";
  var K = {
    // C[M,N] (=|+=) op(A)[M,K] op(B)[K,N] (+ bias[N]) (relu). a M, b N, c K, d lda, e ldb, f ldc, g offA, h offB, i offC,
    // j transA, k transB, l accumulate, m bias, n offBias, o relu, p split: when nonzero, workgroup z sums only K rows
    // [z*p, z*p+p) and writes its partial product at offC + z*M*ldc (summed afterwards by `parts`)
    matmul: U + `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> Bias: array<f32>;
var<workgroup> tA: array<array<f32, 16>, 16>;
var<workgroup> tB: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let row = wg.y * 16u + l.y; let col = wg.x * 16u + l.x;
  var s = 0.0;
  var kLo = 0u; var kHi = u.c;
  if (u.p > 0u) { kLo = wg.z * u.p; kHi = min(u.c, kLo + u.p); }
  for (var t = kLo; t < kHi; t += 16u) {
    let ka = t + l.x;
    var va = 0.0;
    if (row < u.a && ka < kHi) { if (u.j == 1u) { va = A[u.g + ka * u.d + row]; } else { va = A[u.g + row * u.d + ka]; } }
    tA[l.y][l.x] = va;
    let kb = t + l.y;
    var vb = 0.0;
    if (col < u.b && kb < kHi) { if (u.k == 1u) { vb = B[u.h + col * u.e + kb]; } else { vb = B[u.h + kb * u.e + col]; } }
    tB[l.y][l.x] = vb;
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) { s += tA[l.y][k] * tB[k][l.x]; }
    workgroupBarrier();
  }
  if (row < u.a && col < u.b) {
    let i = u.i + wg.z * u.a * u.f + row * u.f + col;
    if (u.m == 1u) { s += Bias[u.n + col]; }
    if (u.l == 1u) { s += C[i]; }
    if (u.o == 1u) { s = max(s, 0.0); }
    C[i] = s;
  }
}`,
    // x[b,t,:] = E[idx[b,t]] + P[t]. a B, b offE, c offP
    embed: U + `
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= u.a * 192u) { return; }
  let c = i % 64u; let bt = i / 64u; let t = bt % 3u;
  x[i] = W[u.b + idx[bt] * 64u + c] + W[u.c + t * 64u + c];
}`,
    // one thread per (row, head): scores, softmax over the 3 positions, the weighted sum of values. a B
    attnF: U + `
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> at: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= u.a * 2u) { return; }
  let b = i / 2u; let lo = (i % 2u) * 32u;
  var sc = array<f32, 3>(0.0, 0.0, 0.0);
  for (var t = 0u; t < 3u; t++) {
    var s = 0.0;
    for (var c = 0u; c < 32u; c++) { s += q[b * 64u + lo + c] * k[(b * 3u + t) * 64u + lo + c]; }
    sc[t] = s * 0.17677669529663687;
  }
  let m = max(sc[0], max(sc[1], sc[2]));
  let e0 = exp(sc[0] - m); let e1 = exp(sc[1] - m); let e2 = exp(sc[2] - m); let z = e0 + e1 + e2;
  let a = array<f32, 3>(e0 / z, e1 / z, e2 / z);
  for (var t = 0u; t < 3u; t++) { at[i * 3u + t] = a[t]; }
  for (var c = 0u; c < 32u; c++) {
    y[b * 64u + lo + c] = a[0] * v[(b * 3u) * 64u + lo + c] + a[1] * v[(b * 3u + 1u) * 64u + lo + c] + a[2] * v[(b * 3u + 2u) * 64u + lo + c];
  }
}`,
    // dst[i,j] (=|+=) src[i,j] over rows x cols. a rows, b cols, c lds, d offS, e ldd, f offD, g accumulate
    copy: U + `
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= u.a * u.b) { return; }
  let r = i / u.b; let c = i % u.b;
  let v = src[u.d + r * u.c + c];
  let o = u.f + r * u.e + c;
  if (u.g == 1u) { dst[o] = dst[o] + v; } else { dst[o] = v; }
}`,
    // softmax cross-entropy per row; logits become d(mean loss)/d(logits). a B, b nout
    softmaxCE: U + `
@group(0) @binding(1) var<storage, read> lab: array<u32>;
@group(0) @binding(2) var<storage, read_write> lg: array<f32>;
@group(0) @binding(3) var<storage, read_write> stat: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x; if (b >= u.a) { return; }
  let o = b * u.b; let y = lab[b];
  var m = lg[o]; var best = 0u;
  for (var j = 1u; j < u.b; j++) { if (lg[o + j] > m) { m = lg[o + j]; best = j; } }
  var z = 0.0;
  for (var j = 0u; j < u.b; j++) { z += exp(lg[o + j] - m); }
  stat[b * 2u] = log(z) + m - lg[o + y];
  stat[b * 2u + 1u] = select(0.0, 1.0, best == y);
  let inv = 1.0 / f32(u.a);
  for (var j = 0u; j < u.b; j++) { lg[o + j] = exp(lg[o + j] - m) / z * inv; }
  lg[o + y] = lg[o + y] - inv;
}`,
    // the index of each row's largest logit. a B, b nout
    argmax: U + `
@group(0) @binding(1) var<storage, read> lg: array<f32>;
@group(0) @binding(2) var<storage, read_write> pred: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x; if (b >= u.a) { return; }
  let o = b * u.b; var m = lg[o]; var best = 0u;
  for (var j = 1u; j < u.b; j++) { if (lg[o + j] > m) { m = lg[o + j]; best = j; } }
  pred[b] = best;
}`,
    // the sums of the per-row loss and correct flags. a B
    reduce: U + `
@group(0) @binding(1) var<storage, read> stat: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> s0: array<f32, 256>;
var<workgroup> s1: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u) {
  var a = 0.0; var c = 0.0;
  for (var i = l.x; i < u.a; i += 256u) { a += stat[i * 2u]; c += stat[i * 2u + 1u]; }
  s0[l.x] = a; s1[l.x] = c;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) {
    if (l.x < w) { s0[l.x] += s0[l.x + w]; s1[l.x] += s1[l.x + w]; }
    workgroupBarrier();
  }
  if (l.x == 0u) { out[0] = s0[0]; out[1] = s1[0]; }
}`,
    // dst[i,j] (=|+=) the sum of a split matmul's partial products. a M, b N, c parts, d ldd, e offD, f accumulate
    parts: U + `
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= u.a * u.b) { return; }
  var s = 0.0;
  for (var z = 0u; z < u.c; z++) { s += src[z * u.a * u.b + i]; }
  let o = u.e + (i / u.b) * u.d + i % u.b;
  if (u.f == 1u) { s += dst[o]; }
  dst[o] = s;
}`,
    // dz = dr where the ReLU was active. a count
    mask: U + `
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= u.a) { return; }
  if (r[i] <= 0.0) { d[i] = 0.0; }
}`,
    // attention backward, one thread per (row, head). a B
    attnB: U + `
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> at: array<f32>;
@group(0) @binding(5) var<storage, read> dy: array<f32>;
@group(0) @binding(6) var<storage, read_write> dq: array<f32>;
@group(0) @binding(7) var<storage, read_write> dk: array<f32>;
@group(0) @binding(8) var<storage, read_write> dv: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= u.a * 2u) { return; }
  let b = i / 2u; let lo = (i % 2u) * 32u;
  let a = array<f32, 3>(at[i * 3u], at[i * 3u + 1u], at[i * 3u + 2u]);
  var da = array<f32, 3>(0.0, 0.0, 0.0);
  for (var t = 0u; t < 3u; t++) {
    var s = 0.0;
    for (var c = 0u; c < 32u; c++) {
      let o = (b * 3u + t) * 64u + lo + c;
      dv[o] = a[t] * dy[b * 64u + lo + c];
      s += dy[b * 64u + lo + c] * v[o];
    }
    da[t] = s;
  }
  let mean = a[0] * da[0] + a[1] * da[1] + a[2] * da[2];
  var ds = array<f32, 3>(0.0, 0.0, 0.0);
  for (var t = 0u; t < 3u; t++) { ds[t] = a[t] * (da[t] - mean) * 0.17677669529663687; }
  for (var c = 0u; c < 32u; c++) {
    var s = 0.0;
    for (var t = 0u; t < 3u; t++) {
      let o = (b * 3u + t) * 64u + lo + c;
      s += ds[t] * k[o];
      dk[o] = ds[t] * q[b * 64u + lo + c];
    }
    dq[b * 64u + lo + c] = s;
  }
}`,
    // the embedding gradient: for each token, the sum of dx over its occurrences (a CSR list). a nin, b offE
    gradE: U + `
@group(0) @binding(1) var<storage, read> start: array<u32>;
@group(0) @binding(2) var<storage, read> occ: array<u32>;
@group(0) @binding(3) var<storage, read> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= u.a * 64u) { return; }
  let tok = i / 64u; let c = i % 64u;
  var s = 0.0;
  for (var o = start[tok]; o < start[tok + 1u]; o++) { s += dx[occ[o] * 64u + c]; }
  G[u.b + i] = s;
}`,
    // Adam, PyTorch's form. a count, b lr, c 1/(1-b1^t), d sqrt(1-b2^t)  (b, c, d are f32 bits)
    adam: U + `
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> M: array<f32>;
@group(0) @binding(4) var<storage, read_write> V: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= u.a) { return; }
  let g = G[i];
  let m = 0.9 * M[i] + 0.1 * g; let v = 0.999 * V[i] + 0.001 * g * g;
  M[i] = m; V[i] = v;
  W[i] = W[i] - bitcast<f32>(u.b) * bitcast<f32>(u.c) * m / (sqrt(v) / bitcast<f32>(u.d) + 1e-8);
}`,
  };

  function TrainGPU(device, L) {
    this.device = device; this.L = L; this.pipes = {}; this.ops = null;
    for (var name in K) {
      var module = device.createShaderModule({ code: K[name] });
      this.pipes[name] = device.createComputePipeline({ layout: "auto", compute: { module: module, entryPoint: "main" } });
    }
    var S = GPUBufferUsage.STORAGE, C = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.buf = function (n, usage) { return device.createBuffer({ size: Math.max(16, Math.ceil(n * 4 / 16) * 16), usage: usage || (S | C) }); };
    this.W = this.buf(L.size); this.G = this.buf(L.size); this.M = this.buf(L.size); this.V = this.buf(L.size);
    this.adamU = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.out = this.buf(2); this.read2 = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  TrainGPU.create = async function (L) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
    var adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    var device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
    var g = new TrainGPU(device, L);
    g.adapterInfo = adapter.info || {};
    return g;
  };

  TrainGPU.prototype.uniform = function (vals, floats) {
    var a = new Uint32Array(16), f = new Float32Array(a.buffer);
    vals.forEach(function (v, i) { if (floats && floats.indexOf(i) >= 0) f[i] = v; else a[i] = v; });
    var b = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, a);
    return b;
  };
  TrainGPU.prototype.op = function (list, name, vals, buffers, groups) {
    var entries = [{ binding: 0, resource: { buffer: this.uniform(vals) } }];
    buffers.forEach(function (b, i) { entries.push({ binding: i + 1, resource: { buffer: b } }); });
    list.push({ pipe: this.pipes[name], group: this.device.createBindGroup({ layout: this.pipes[name].getBindGroupLayout(0), entries: entries }), n: groups });
  };

  // The passes for `rows` examples whose token ids and labels are in idx/lab: forward, then (training) the loss and
  // backward, or (evaluation) the argmax.
  TrainGPU.prototype.passes = function (rows, idx, lab, csr, train) {
    var L = this.L, o = L.off, B = rows, a = this.act, W = this.W, G = this.G, ops = [], self = this;
    var wg = function (n, size) { return [Math.ceil(n / size)]; };
    // A product whose inner dimension is the batch (a weight gradient) is split into chunks of SPLIT rows, run in
    // parallel, then summed: otherwise a few workgroups would each walk tens of thousands of rows.
    function mm(M, N, Kk, A, lda, offA, transA, Bb, ldb, offB, transB, Cc, ldc, offC, opts) {
      opts = opts || {};
      if (Kk > 2 * SPLIT) {
        var parts = Math.ceil(Kk / SPLIT);
        self.op(ops, "matmul", [M, N, Kk, lda, ldb, N, offA, offB, 0, transA, transB, 0, 0, 0, 0, SPLIT], [A, Bb, self.scratch, W],
          [Math.ceil(N / 16), Math.ceil(M / 16), parts]);
        self.op(ops, "parts", [M, N, parts, ldc, offC, opts.acc ? 1 : 0], [self.scratch, Cc], wg(M * N, 256));
        return;
      }
      self.op(ops, "matmul", [M, N, Kk, lda, ldb, ldc, offA, offB, offC, transA, transB, opts.acc ? 1 : 0, opts.bias !== undefined ? 1 : 0,
        opts.bias || 0, opts.relu ? 1 : 0, 0], [A, Bb, Cc, W], [Math.ceil(N / 16), Math.ceil(M / 16)]);
    }
    // the column sums of a rows x cols block (a bias gradient): ones^T times the block
    function colsum(rowsN, cols, src, ld, off, offDst) { mm(1, cols, rowsN, self.ones, 1, 0, 1, src, ld, off, 0, G, cols, offDst); }
    function copy(rowsN, cols, src, lds, offS, dst, ldd, offD, acc) {
      self.op(ops, "copy", [rowsN, cols, lds, offS, ldd, offD, acc ? 1 : 0], [src, dst], wg(rowsN * cols, 256));
    }
    this.op(ops, "embed", [B, o.E.at, o.P.at], [idx, W, a.x], wg(B * 192, 256));
    mm(B, D, D, a.x, 192, 128, 0, W, D, o.Wq.at, 1, a.q, D, 0);
    mm(3 * B, D, D, a.x, D, 0, 0, W, D, o.Wk.at, 1, a.k, D, 0);
    mm(3 * B, D, D, a.x, D, 0, 0, W, D, o.Wv.at, 1, a.v, D, 0);
    this.op(ops, "attnF", [B], [a.q, a.k, a.v, a.at, a.y], wg(B * 2, 64));
    copy(B, D, a.x, 192, 128, a.h1, D, 0);
    mm(B, D, D, a.y, D, 0, 0, W, D, o.Wo.at, 1, a.h1, D, 0, { acc: 1 });
    mm(B, HID, D, a.h1, D, 0, 0, W, D, o.W1.at, 1, a.r, HID, 0, { bias: o.b1.at, relu: 1 });
    copy(B, D, a.h1, D, 0, a.h2, D, 0);
    mm(B, D, HID, a.r, HID, 0, 0, W, HID, o.W2.at, 1, a.h2, D, 0, { acc: 1, bias: o.b2.at });
    mm(B, L.nout, D, a.h2, D, 0, 0, W, D, o.U.at, 1, a.lg, L.nout, 0, { bias: o.bU.at });
    if (!train) { this.op(ops, "argmax", [B, L.nout], [a.lg, a.pred], wg(B, 64)); return ops; }
    this.op(ops, "softmaxCE", [B, L.nout], [lab, a.lg, a.stat], wg(B, 64));
    this.op(ops, "reduce", [B], [a.stat, this.out], [1]);
    mm(L.nout, D, B, a.lg, L.nout, 0, 1, a.h2, D, 0, 0, G, D, o.U.at);                       // dU = dlogits^T h2
    colsum(B, L.nout, a.lg, L.nout, 0, o.bU.at);
    mm(B, D, L.nout, a.lg, L.nout, 0, 0, W, D, o.U.at, 0, a.dh2, D, 0);                       // dh2 = dlogits U
    mm(D, HID, B, a.dh2, D, 0, 1, a.r, HID, 0, 0, G, HID, o.W2.at);                           // dW2 = dh2^T r
    colsum(B, D, a.dh2, D, 0, o.b2.at);
    mm(B, HID, D, a.dh2, D, 0, 0, W, HID, o.W2.at, 0, a.dz, HID, 0);                          // dr = dh2 W2
    this.op(ops, "mask", [B * HID], [a.r, a.dz], wg(B * HID, 256));                            // dz
    mm(HID, D, B, a.dz, HID, 0, 1, a.h1, D, 0, 0, G, D, o.W1.at);                             // dW1 = dz^T h1
    colsum(B, HID, a.dz, HID, 0, o.b1.at);
    copy(B, D, a.dh2, D, 0, a.dh1, D, 0);
    mm(B, D, HID, a.dz, HID, 0, 0, W, D, o.W1.at, 0, a.dh1, D, 0, { acc: 1 });                // dh1 = dh2 + dz W1
    mm(D, D, B, a.dh1, D, 0, 1, a.y, D, 0, 0, G, D, o.Wo.at);                                 // dWo = dh1^T y
    mm(B, D, D, a.dh1, D, 0, 0, W, D, o.Wo.at, 0, a.dy, D, 0);                                // dy = dh1 Wo
    this.op(ops, "attnB", [B], [a.q, a.k, a.v, a.at, a.dy, a.dq, a.dk, a.dv], wg(B * 2, 64));
    mm(D, D, B, a.dq, D, 0, 1, a.x, 192, 128, 0, G, D, o.Wq.at);                              // dWq = dq^T x2
    mm(D, D, 3 * B, a.dk, D, 0, 1, a.x, D, 0, 0, G, D, o.Wk.at);                              // dWk = dk^T x
    mm(D, D, 3 * B, a.dv, D, 0, 1, a.x, D, 0, 0, G, D, o.Wv.at);                              // dWv = dv^T x
    mm(3 * B, D, D, a.dk, D, 0, 0, W, D, o.Wk.at, 0, a.dx, D, 0);                             // dx = dk Wk
    mm(3 * B, D, D, a.dv, D, 0, 0, W, D, o.Wv.at, 0, a.dx, D, 0, { acc: 1 });                 //    + dv Wv
    mm(B, D, D, a.dq, D, 0, 0, W, D, o.Wq.at, 0, a.dx, 192, 128, { acc: 1 });                 // dx2 += dq Wq
    copy(B, D, a.dh1, D, 0, a.dx, 192, 128, true);                                            // dx2 += dh1
    for (var t = 0; t < 3; t++) colsum(B, D, a.dx, 192, t * D, o.P.at + t * D);
    this.op(ops, "gradE", [L.nin, o.E.at], [csr.start, csr.occ, a.dx, G], wg(L.nin * D, 64));
    return ops;
  };

  // Load the whole table (for evaluation) and a training set (indices into the table).
  TrainGPU.prototype.setData = function (table, train) {
    var L = this.L, n = table.labels.length, dev = this.device, self = this;
    if (this.act) Object.keys(this.act).forEach(function (k) { self.act[k].destroy(); });
    var B = Math.max(n, train.length), f = function (k) { return self.buf(k); };
    if (this.scratch) { this.scratch.destroy(); this.ones.destroy(); }
    this.scratch = f(Math.ceil(3 * B / SPLIT) * L.nout * D);
    this.ones = f(B); dev.queue.writeBuffer(this.ones, 0, new Float32Array(B).fill(1));
    this.act = { x: f(B * 192), q: f(B * D), k: f(B * 192), v: f(B * 192), at: f(B * 6), y: f(B * D), h1: f(B * D), r: f(B * HID),
                 h2: f(B * D), lg: f(B * L.nout), stat: f(B * 2), pred: f(B), dh2: f(B * D), dz: f(B * HID), dh1: f(B * D),
                 dy: f(B * D), dq: f(B * D), dk: f(B * 192), dv: f(B * 192), dx: f(B * 192) };
    function upload(arr) { var b = self.buf(arr.length); dev.queue.writeBuffer(b, 0, arr); self.owned.push(b); return b; }
    (this.owned || []).forEach(function (b) { b.destroy(); });
    this.owned = [];
    var allIdx = upload(Uint32Array.from(table.steps)), allLab = upload(Uint32Array.from(table.labels));
    var tIdx = new Uint32Array(train.length * 3), tLab = new Uint32Array(train.length), count = new Uint32Array(L.nin + 1);
    train.forEach(function (k, i) {
      for (var t = 0; t < 3; t++) { tIdx[3 * i + t] = table.steps[3 * k + t]; count[tIdx[3 * i + t] + 1]++; }
      tLab[i] = table.labels[k];
    });
    for (var i = 0; i < L.nin; i++) count[i + 1] += count[i];
    var occ = new Uint32Array(train.length * 3), fillAt = count.slice(0, L.nin);
    for (i = 0; i < tIdx.length; i++) occ[fillAt[tIdx[i]]++] = i;
    var csr = { start: upload(count), occ: upload(occ) };
    this.nTrain = train.length; this.nAll = n;
    this.trainOps = this.passes(train.length, upload(tIdx), upload(tLab), csr, true);
    this.evalOps = this.passes(n, allIdx, allLab, null, false);
  };

  TrainGPU.prototype.setParams = function (p) {
    var q = this.device.queue;
    q.writeBuffer(this.W, 0, p); q.writeBuffer(this.M, 0, new Float32Array(this.L.size)); q.writeBuffer(this.V, 0, new Float32Array(this.L.size));
    q.writeBuffer(this.G, 0, new Float32Array(this.L.size));
    this.t = 0;
  };

  TrainGPU.prototype.encode = function (enc, ops) {
    var pass = enc.beginComputePass();
    ops.forEach(function (op) { pass.setPipeline(op.pipe); pass.setBindGroup(0, op.group); pass.dispatchWorkgroups(op.n[0], op.n[1] || 1, op.n[2] || 1); });
    pass.end();
  };

  // One full-batch training step (gradient, then Adam). Resolves to {loss: mean loss, correct: training examples right},
  // both measured before the update.
  TrainGPU.prototype.step = async function (lr, skipAdam) {
    var dev = this.device, enc = dev.createCommandEncoder();
    this.encode(enc, this.trainOps);
    if (!skipAdam) {
      this.t++;
      var a = new Uint32Array(16), f = new Float32Array(a.buffer);
      a[0] = this.L.size; f[1] = lr; f[2] = 1 / (1 - Math.pow(0.9, this.t)); f[3] = Math.sqrt(1 - Math.pow(0.999, this.t));
      dev.queue.writeBuffer(this.adamU, 0, a);
      if (!this.adamOp) {
        this.adamOp = [{ pipe: this.pipes.adam, n: [Math.ceil(this.L.size / 256)], group: dev.createBindGroup({ layout: this.pipes.adam.getBindGroupLayout(0),
          entries: [this.adamU, this.G, this.W, this.M, this.V].map(function (b, i) { return { binding: i, resource: { buffer: b } }; }) }) }];
      }
      this.encode(enc, this.adamOp);
    }
    enc.copyBufferToBuffer(this.out, 0, this.read2, 0, 8);
    dev.queue.submit([enc.finish()]);
    await this.read2.mapAsync(GPUMapMode.READ);
    var r = new Float32Array(this.read2.getMappedRange().slice(0));
    this.read2.unmap();
    return { loss: r[0] / this.nTrain, correct: Math.round(r[1]) };
  };

  TrainGPU.prototype.read = async function (buf, n, Type) {
    var dev = this.device, st = dev.createBuffer({ size: Math.ceil(n * 4 / 16) * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    var enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, st, 0, Math.ceil(n * 4 / 16) * 16); dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    var out = new (Type || Float32Array)(st.getMappedRange().slice(0, n * 4));
    st.unmap(); st.destroy();
    return out;
  };
  // The network's prediction for every step of the table.
  TrainGPU.prototype.predictAll = async function () {
    var enc = this.device.createCommandEncoder(); this.encode(enc, this.evalOps); this.device.queue.submit([enc.finish()]);
    return this.read(this.act.pred, this.nAll, Int32Array);
  };
  TrainGPU.prototype.params = function () { return this.read(this.W, this.L.size); };
  TrainGPU.prototype.grads = function () { return this.read(this.G, this.L.size); };

  root.TrainGPU = TrainGPU;
})(this);
