// One forward pass of the version 2 network (data2.js) in 3D, shared by inside.html (the explorer) and walk.html (the
// walkthrough you scroll). The values drawn are the ones Network.prototype.forward (core.js) returns, the same function
// the decoder runs; the window comes live from a run of the sieve of Eratosthenes to 120, or is one of the three it
// reads that were never in training (inside-data.js).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

export const D = window.TCT_DATA2, TCT = window.TCT, INSIDE = window.TCT_INSIDE, W = D.weights;
export const engine = new TCT.Engine(D), vin = D.vin, vout = D.vout;
const net = engine.net;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------------------------------------------------------------- words and numbers

export const comma = (n) => n.toLocaleString("en-US");
export const f3 = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(3));
export const pct = (x) => Math.round(x * 100) + "%";
export const POS = ["l", "c", "r"];
export function fmtP(p) { return p > 0.9995 ? "1.000" : p < 0.0005 ? (p < 1e-6 ? "< 0.000001" : p.toFixed(6)) : p.toFixed(3); }
export const code = (t) => "<code>" + t.replace(/&/g, "&amp;").replace(/</g, "&lt;") + "</code>";
export function describe(t) {
  if (t === "#") return "the ring's start";
  const i = t.indexOf(":");
  if (i < 0) return "a cell holding " + code(t);
  const st = t.slice(0, i);
  return "the universal machine's head, in state " + code(st) + " on " + code(t.slice(i + 1)) + " (" + TCT.phase(st) + ")";
}
export const maxAbs = (x) => { let m = 0; for (const v of x) m = Math.max(m, Math.abs(v)); return m; };

// ---------------------------------------------------------------- windows: live from the sieve, or one it never saw

export function windows() {
  const SIEVE = window.TCT_PRIMES.sieve(120), machine = TCT.parseMachine(SIEVE.start, SIEVE.rules);
  const run = new TCT.Run(engine, machine, SIEVE.tape), BID = engine.id["#"];
  let readStart = 1;                                     // where the ring being read starts in run.T
  const isHead = (id) => vin[id].indexOf(":") >= 0;
  function step() { run.step(false); if (run.T[run.k + 1] === BID) readStart = run.k + 1; }
  function live() {
    const k = run.k, ids = run.window(), tok = (a, b) => run.tokens(a, b);
    let end = k + 2; while (end < run.n && run.T[end] !== BID) end++;
    const written = run.ringStart === null || vin[ids[1]] === "#" ? [] : tok(Math.max(run.ringStart + 1, run.n - 12), run.n);
    return { ids, live: true, pass: run.passes + 1, ring: run.utmSteps, at: k + 1 - readStart, ringLen: end - readStart,
             before: tok(Math.max(readStart, k - 12), k), after: tok(k + 3, Math.min(end, k + 15)), written,
             writtenMore: run.ringStart !== null && run.n - 12 > run.ringStart + 1 };
  }
  function nextHead() { let n = 0; do { step(); n++; } while (!run.window().some(isHead) && n < 200000 && !run.verdict); }
  function unseen(i) {
    const u = INSIDE.unseen[i];
    return { ids: u.window.map((t) => engine.id[t]), live: false, unseen: i, pass: u.pass + 1, ring: u.utmStep, at: u.at,
             ringLen: u.ring, before: u.before.slice(-12), after: u.after.slice(0, 12), written: null };
  }
  // the first pass with the head in the middle, two rings in
  function start() { while (run.utmSteps < 2 || !isHead(run.window()[1])) step(); return live(); }
  return { run, step, live, nextHead, unseen, start, isHead };
}

// Run the pass on a window, and keep what the pages show.
export function analyse(v) {
  const f = net.forward(v.ids), lg = f.logits;
  let mx = -Infinity; for (const x of lg) mx = Math.max(mx, x);
  let z = 0; for (const x of lg) z += Math.exp(x - mx);
  const order = Array.from(lg.keys()).sort((a, b) => lg[b] - lg[a]);
  v.f = f; v.order = order; v.prob = (i) => Math.exp(lg[i] - mx) / z;
  v.rank = new Int32Array(lg.length); order.forEach((i, r) => { v.rank[i] = r; });
  v.out = D.emits[vout[f.best]]; v.exact = engine.exact(v.ids); v.trained = engine.trained.has(engine.key(v.ids));
  v.wrote = v.out.join(" ");
  return v;
}

// The transcript around the window, as two lines of tokens: the ring being read, and the ring being written.
export function drawStrip(el, v, revealed) {
  el.innerHTML = "";
  const tok = (t, cls, role) => { const s = document.createElement("span");
    s.className = "tok" + (t.indexOf(":") >= 0 ? " head" : "") + (cls ? " " + cls : ""); s.textContent = t; if (role) s.dataset.r = role; return s; };
  const line = (label) => { const d = document.createElement("div"); d.className = "line"; const l = document.createElement("span");
    l.className = "lab"; l.textContent = label; d.appendChild(l); el.appendChild(d); return d; };
  const r = line("reading ring " + comma(v.ring));
  if (v.before.length < v.at - 1) r.appendChild(tok("…", "more"));
  v.before.forEach((t) => r.appendChild(tok(t)));
  ["l", "c", "r"].forEach((role, i) => r.appendChild(tok(vin[v.ids[i]], "win w-" + role, role)));
  v.after.forEach((t) => r.appendChild(tok(t)));
  if (v.at + 2 + v.after.length < v.ringLen) r.appendChild(tok("…", "more"));
  const w = line(vin[v.ids[1]] === "#" ? "starts ring " + comma(v.ring + 1) : "writing ring " + comma(v.ring + 1));
  if (v.written) { if (v.writtenMore) w.appendChild(tok("…", "more")); v.written.forEach((t) => w.appendChild(tok(t))); }
  else w.appendChild(tok("…", "more"));
  (v.out.length ? v.out : ["(halt)"]).forEach((t) => { const s = tok("?", "out"); s.dataset.t = t; w.appendChild(s); });
  revealStrip(el, revealed);
}
export function revealStrip(el, on) {
  el.querySelectorAll(".tok.out").forEach((s) => { s.textContent = on ? s.dataset.t : "?"; s.classList.toggle("shown", on); });
}

// Where the 23,844 parameters are.
export function budgetTable(el) {
  const flat = (M) => [].concat(...(Array.isArray(M[0]) ? M : [M]));
  const n = (k) => flat(W[k]).length;
  const rows = [
    ["E", "embedding table", vin.length + " × 8", n("E.weight")],
    ["P", "positions", "3 × 8", n("P")],
    ["Wq, Wk, Wv, Wo", "attention", "4 × (8 × 8)", n("Wq.weight") + n("Wk.weight") + n("Wv.weight") + n("Wo.weight")],
    ["W1, W2 and biases", "the MLP", "8 → 32 → 8", n("W1.weight") + n("W1.bias") + n("W2.weight") + n("W2.bias")],
    ["U and bias", "unembedding", vout.length + " × 8 + " + vout.length, n("U.weight") + n("U.bias")],
  ];
  const total = rows.reduce((s, r) => s + r[3], 0);
  el.innerHTML = "<thead><tr><th>Block</th><th>What it is</th><th>Shape</th><th class='n'>Parameters</th><th class='n'>Share</th></tr></thead><tbody>" +
    rows.map((r) => "<tr><td>" + r[0] + "</td><td>" + r[1] + "</td><td class='mono'>" + r[2] + "</td><td class='n mono'>" + comma(r[3]) +
      "</td><td class='n mono'>" + (r[3] / total * 100).toFixed(1) + "%</td></tr>").join("") +
    "</tbody><tfoot><tr><td colspan='3'>Total</td><td class='n mono'>" + comma(total) + "</td><td class='n mono'>100%</td></tr></tfoot>";
  if (total !== D.params) console.warn("parameter count", total, "differs from", D.params);
}

// ---------------------------------------------------------------- the stages of the pass, and where the camera looks

export const R2 = 10;                                      // the scene's second row: the MLP, the unembedding, the probabilities
export const CH = [
  { title: "The window", view: [[-7.4, 0, 5.4], [-7.4, 23.5, 27]] },
  { title: "Look up the tokens", view: [[-13.4, 0, 0.3], [-13.4, 10.5, 11.6]] },
  { title: "Queries, keys and values", view: [[-3.8, 0, 0.3], [-3.8, 7, 7.4]] },
  { title: "Attention", view: [[-1, 0, 0.6], [-0.4, 6.4, 6.9]] },
  { title: "Back into the stream", view: [[3.6, 0, 1.4], [3.6, 6, 7]] },
  { title: "The MLP", view: [[-15.8, 0, R2 + 0.6], [-15.8, 7.2, R2 + 7.6]] },
  { title: "Scoring every output", view: [[-1.6, 0, R2], [-1.6, 10, R2 + 10.5]] },
  { title: "What it writes", view: [[3.2, 0.7, R2], [3.2, 3.6, R2 + 7.6]] },
];
export const ALL = 8;                                      // a stage past the last: the whole pass, every value shown

// ---------------------------------------------------------------- the scene

// el: the stage element; tip: an element for hover values. opts.controls: let the viewer turn and zoom the view.
export function createScene(el, tip, opts = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  el.appendChild(renderer.domElement);
  const labels = new CSS2DRenderer(); labels.domElement.className = "labels3d"; el.appendChild(labels.domElement);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.1, 400);
  camera.position.set(...CH[0].view[1]);
  const target = new THREE.Vector3(...CH[0].view[0]);
  let controls = null;
  if (opts.controls) {
    controls = new OrbitControls(camera, labels.domElement);
    controls.enableDamping = true; controls.target.copy(target); controls.maxPolarAngle = Math.PI * 0.48;
    controls.addEventListener("start", () => { camTween = null; });
    // zoom only with a pinch or ⌘/Ctrl + scroll, and never past the camera's far plane (400)
    controls.enableZoom = false; controls.maxDistance = 300;
    el.addEventListener("wheel", (e) => { controls.enableZoom = e.ctrlKey || e.metaKey; }, { capture: true, passive: true });
  } else camera.lookAt(target);
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(-4, 10, 6); scene.add(sun);
  function resize() {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h, false); labels.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(el); resize();
  let visible = true;
  new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(el);
  let hooks = [];

  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const isDark = () => getComputedStyle(document.body).backgroundColor.match(/\d+/g).slice(0, 3).reduce((a, b) => a + +b, 0) < 300;
  const COL = { pos: new THREE.Color(), neg: new THREE.Color(), mid: new THREE.Color(), h1: new THREE.Color(), h2: new THREE.Color(),
                plate: new THREE.Color(), hi: new THREE.Color() };
  function theme() {
    const dark = isDark();
    COL.pos.set(dark ? "#d95926" : "#eb6834"); COL.neg.set(dark ? "#3987e5" : "#2a78d6");
    COL.mid.set(dark ? "#4a4f59" : "#c3c2b7"); COL.h1.set(dark ? "#199e70" : "#1baf7a"); COL.h2.set(dark ? "#9a72f0" : "#8b5cf6");
    COL.plate.set(css("--rule") || "#e2e0d9"); COL.hi.set(dark ? "#f2c94c" : "#e0a400");
  }
  theme();

  const BOX = new THREE.BoxGeometry(1, 1, 1); BOX.translate(0, 0.5, 0);
  const blocks = [], hoverable = [], tmpM = new THREE.Matrix4(), tmpC = new THREE.Color();
  let view = null, cur = 0, flyToken = 0;

  // A block of bars: n numbers at fixed places on the floor, each a bar whose height is the number's size.
  class Bars {
    constructor(o) {
      Object.assign(this, { cell: 0.2, H: 1, chapter: 0, animate: true, color: null }, o);
      const n = this.n;
      this.mat = new THREE.MeshLambertMaterial({ transparent: true, opacity: 1 });
      this.mesh = new THREE.InstancedMesh(BOX, this.mat, n);
      this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.mesh.userData.block = this;
      this.group = new THREE.Group(); this.group.position.set(this.x, 0, this.z); this.group.add(this.mesh);
      scene.add(this.group);
      this.px = new Float32Array(n); this.pz = new Float32Array(n);
      for (let i = 0; i < n; i++) { const p = this.place(i); this.px[i] = p[0]; this.pz[i] = p[1]; }
      this.target = new Float32Array(n); this.cur = new Float32Array(n); this.show = this.weights ? 1 : 0; this.vis = this.show;
      this.norm = 1; this.dirty = true; this.opacity = 1;
      let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
      for (let i = 0; i < n; i++) { x0 = Math.min(x0, this.px[i]); x1 = Math.max(x1, this.px[i]); z0 = Math.min(z0, this.pz[i]); z1 = Math.max(z1, this.pz[i]); }
      this.box = [x0 - this.cell, x1 + this.cell, z0 - this.cell, z1 + this.cell];
      const plate = new THREE.Mesh(new THREE.BoxGeometry(this.box[1] - this.box[0], 0.02, this.box[3] - this.box[2]),
                                   new THREE.MeshLambertMaterial({ color: COL.plate, transparent: true, opacity: 0.9 }));
      plate.position.set((this.box[0] + this.box[1]) / 2, -0.012, (this.box[2] + this.box[3]) / 2);
      this.plate = plate; this.group.add(plate);
      if (this.label) {
        const o = this.addLabel(this.label, (this.box[0] + this.box[1]) / 2, this.labelFront ? this.box[3] + 0.08 : this.box[2] - 0.08);
        if (this.labelFront) o.center.set(0.5, 0);
      }
      blocks.push(this); hoverable.push(this.mesh);
    }
    addLabel(text, x, z, cls) {
      const d = document.createElement("div"); d.className = "v-label small" + (cls ? " " + cls : ""); d.textContent = text;
      const o = new CSS2DObject(d); o.position.set(x, 0.05, z); o.center.set(0.5, 1); this.group.add(o);
      (this.labels = this.labels || []).push(o);
      return o;
    }
    set(values, norm) {
      for (let i = 0; i < this.n; i++) this.target[i] = values[i];
      this.norm = norm || maxAbs(this.target) || 1; this.dirty = true;
      if (!this.animate) this.cur.set(this.target);
    }
    colorOf(i, v, out) {
      if (this.color) return this.color(i, v, out);
      const t = Math.min(1, Math.abs(v) / this.norm);
      return out.copy(COL.mid).lerp(v >= 0 ? COL.pos : COL.neg, 0.3 + 0.7 * Math.sqrt(t));
    }
    frame(dt) {
      const k = this.animate && !reduced ? 1 - Math.exp(-dt * 5) : 1;
      let changed = this.dirty;
      if (this.vis !== this.show) {
        changed = true; const nv = this.vis + (this.show - this.vis) * k;
        this.vis = Math.abs(nv - this.show) < 1e-3 ? this.show : nv;
      }
      for (let i = 0; i < this.n; i++) {
        if (this.cur[i] === this.target[i]) continue;
        changed = true; const c = this.cur[i] + (this.target[i] - this.cur[i]) * k;
        this.cur[i] = Math.abs(c - this.target[i]) < 1e-4 * this.norm ? this.target[i] : c;
      }
      if (!changed) return;
      this.dirty = false;
      const w = this.cell * 0.82;
      for (let i = 0; i < this.n; i++) {
        const h = Math.max(0.004, Math.abs(this.cur[i]) / this.norm * this.H * this.vis);
        tmpM.makeScale(w, this.vis > 0.002 ? h : 1e-5, w); tmpM.setPosition(this.px[i], 0, this.pz[i]);
        this.mesh.setMatrixAt(i, tmpM);
        this.mesh.setColorAt(i, this.colorOf(i, this.target[i], tmpC));
      }
      this.mesh.instanceMatrix.needsUpdate = true; if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
      this.mesh.computeBoundingSphere();
    }
    setOpacity(o) {
      this.opacity = o; this.mat.opacity = o; this.mat.depthWrite = o > 0.99; this.plate.material.opacity = 0.9 * Math.max(0.35, o);
    }
  }

  // A vector of n numbers as a row along x.
  const row = (n, cell) => (i) => [(i - (n - 1) / 2) * cell, 0];
  // A matrix (rows x cols): its columns (inputs) along x and its rows (outputs) along z, or the other way round.
  const grid = (rows, cols, cell, transpose) => (i) => { const r = Math.floor(i / cols), c = i % cols;
    return transpose ? [(r - (rows - 1) / 2) * cell, (c - (cols - 1) / 2) * cell] : [(c - (cols - 1) / 2) * cell, (r - (rows - 1) / 2) * cell]; };
  // A lookup table: one tile of 8 bars per token, PER tiles to a row.
  const TC = 0.045, PER = 18;
  const tiles = (count) => { const rows = Math.ceil(count / PER);
    return (i) => { const t = Math.floor(i / 8), j = i % 8, c = t % PER, r = Math.floor(t / PER);
      return [((c - PER / 2) * 9 + j + 0.5) * TC, (r - rows / 2) * 2 * TC]; }; };
  const tileCenter = (block, t) => { const p = block.place(t * 8); return new THREE.Vector3(block.x + p[0] + 3.5 * TC, 0, block.z + p[1]); };
  const flat = (M) => [].concat(...M);

  const VC = 0.2, WC = 0.16, NU = W["W1.weight"].length, QX = -2.4, LG = 37, LC = 0.14;
  const L = {};                                            // the blocks, by name
  // the lookup table E and the positions P (stage 1)
  L.E = new Bars({ n: vin.length * 8, x: -16, z: 0, cell: TC, H: 0.35, chapter: 1, weights: true, animate: false, place: tiles(vin.length), labelFront: true,
                   label: "E · embedding table · 1,351 tokens × 8", info: (i) => "E[" + vin[Math.floor(i / 8)] + "][" + i % 8 + "] = " + f3(W["E.weight"][Math.floor(i / 8)][i % 8]) });
  L.P = new Bars({ n: 24, x: -8.6, z: -2.4, cell: VC, H: 0.6, chapter: 1, weights: true, animate: false, place: grid(3, 8, VC),
                   label: "P · positions · 3 × 8", info: (i) => "P[" + POS[Math.floor(i / 8)] + "][" + i % 8 + "] = " + f3(W.P[Math.floor(i / 8)][i % 8]) });
  L.X = [0, 1, 2].map((p) => new Bars({ n: 8, x: -8.6, z: -0.55 + 0.55 * p, cell: VC, H: 0.9, chapter: 1, place: row(8, VC),
                   info: (i) => "x_" + POS[p] + "[" + i + "] = " + f3(view.f.x[p][i]) }));
  // queries, keys and values (stage 2)
  function weightBlock(name, key, x, z, rows, cols, chapter, label, transpose) {
    const M = W[key];
    const b = new Bars({ n: rows * cols, x, z, cell: WC, H: 0.55, chapter, weights: true, animate: false, place: grid(rows, cols, WC, transpose), label,
      info: (i) => name + "[" + Math.floor(i / cols) + "][" + i % cols + "] = " + f3(M[Math.floor(i / cols)][i % cols]) });
    b.set(flat(M)); return b;
  }
  L.Wq = weightBlock("Wq", "Wq.weight", -5.2, -2.6, 8, 8, 2, "Wq · 8 × 8");
  L.Wk = weightBlock("Wk", "Wk.weight", -5.2, 0, 8, 8, 2, "Wk · 8 × 8");
  L.Wv = weightBlock("Wv", "Wv.weight", -5.2, 2.6, 8, 8, 2, "Wv · 8 × 8");
  const head = (i) => " (head " + (i < 4 ? 1 : 2) + ")";
  L.q = new Bars({ n: 8, x: QX, z: -2.6, cell: VC, H: 0.9, chapter: 2, place: row(8, VC), label: "q · r's query", info: (i) => "q[" + i + "] = " + f3(view.f.q[i]) + head(i) });
  L.K = [0, 1, 2].map((p) => new Bars({ n: 8, x: QX, z: -0.55 + 0.55 * p, cell: VC, H: 0.9, chapter: 2, place: row(8, VC),
    info: (i) => "k_" + POS[p] + "[" + i + "] = " + f3(view.f.k[p][i]) + head(i) }));
  L.V = [0, 1, 2].map((p) => new Bars({ n: 8, x: QX, z: 2.05 + 0.55 * p, cell: VC, H: 0.9, chapter: 2, place: row(8, VC),
    info: (i) => "v_" + POS[p] + "[" + i + "] = " + f3(view.f.v[p][i]) + head(i) }));
  // attention (stage 3)
  L.A = new Bars({ n: 6, x: 0.6, z: 0, cell: 0.3, H: 1.3, chapter: 3, place: (i) => [((i % 3) - 1) * 0.34, (Math.floor(i / 3) - 0.5) * 0.5],
    label: "attention weights", info: (i) => "head " + (Math.floor(i / 3) + 1) + " on " + POS[i % 3] + ": " + view.f.attention[Math.floor(i / 3)][i % 3].toFixed(3),
    color: (i, v, out) => out.copy(COL.mid).lerp(i < 3 ? COL.h1 : COL.h2, 0.25 + 0.75 * v) });
  L.y = new Bars({ n: 8, x: 0.6, z: 2.6, cell: VC, H: 0.9, chapter: 3, place: row(8, VC), label: "y · the heads' mix of values", info: (i) => "y[" + i + "] = " + f3(view.f.y[i]) });
  // back into the stream (stage 4)
  L.Wo = weightBlock("Wo", "Wo.weight", 3, 2.6, 8, 8, 4, "Wo · 8 × 8");
  L.o = new Bars({ n: 8, x: 5.3, z: 2.6, cell: VC, H: 0.9, chapter: 4, place: row(8, VC), label: "o · attention's update", info: (i) => "o[" + i + "] = " + f3(view.f.o[i]) });
  L.h1 = new Bars({ n: 8, x: 5.3, z: 0, cell: VC, H: 0.9, chapter: 4, place: row(8, VC), label: "h = x_r + o", info: (i) => "h[" + i + "] = " + f3(view.f.h1[i]) });
  // the MLP (stage 5), on the second row: W1's inputs along x and its 32 units along z; W2 the other way round
  L.W1 = weightBlock("W1", "W1.weight", -17.2, R2, NU, 8, 5, "W1 · 32 × 8");
  L.hid = new Bars({ n: NU, x: -15.95, z: R2, cell: WC, H: 0.9, chapter: 5, place: (i) => [0, (i - (NU - 1) / 2) * WC], label: "ReLU",
    info: (i) => "unit " + i + ": " + f3(view.f.pre[i]) + (view.f.pre[i] > 0 ? ", on" : ", off (ReLU makes it 0)") });
  L.W2 = weightBlock("W2", "W2.weight", -14.6, R2, 8, NU, 5, "W2 · 8 × 32", true);
  L.h2 = new Bars({ n: 8, x: -14.6, z: R2 + 3.2, cell: VC, H: 0.9, chapter: 5, place: row(8, VC), label: "h + the MLP's update", info: (i) => "h[" + i + "] = " + f3(view.f.h2[i]) });
  // the unembedding and the probabilities (stage 6)
  L.U = new Bars({ n: vout.length * 8, x: -6, z: R2, labelFront: true, cell: TC, H: 0.35, chapter: 6, weights: true, animate: false, place: tiles(vout.length),
                   label: "U · unembedding · 1,356 outputs × 8", info: (i) => "U[" + vout[Math.floor(i / 8)] + "][" + i % 8 + "] = " + f3(W["U.weight"][Math.floor(i / 8)][i % 8]) });
  L.logit = new Bars({ n: vout.length, x: 3.2, z: R2, cell: LC, H: 1.8, chapter: 6, labelFront: true, place: (i) => [((i % LG) - (LG - 1) / 2) * LC, (Math.floor(i / LG) - (LG - 1) / 2) * LC],
    label: "a probability for each of the 1,356 outputs", info: (i) => "“" + vout[i] + "”: score " + f3(view.f.logits[i]) + ", probability " + fmtP(view.prob(i)),
    color: (i, v, out) => i === view.f.best ? out.copy(COL.pos) : view.rank[i] < 3 ? out.copy(COL.mid).lerp(COL.pos, 0.45) : out.copy(COL.mid) });
  L.E.set(flat(W["E.weight"])); L.U.set(flat(W["U.weight"])); L.P.set(flat(W.P));

  // what the parts are, for the overview
  const sections = [["The tokens' rows", -16, -3.7], ["Attention", -3, -3.7], ["The MLP", -15.9, R2 - 3], ["The unembedding", -6, R2 - 3.7],
    ["The probabilities", 3.2, R2 - 3]].map(([t, x, z]) => {
    const d = document.createElement("div"); d.className = "v-label sec"; d.textContent = t;
    const o = new CSS2DObject(d); o.position.set(x, 0, z); o.center.set(0.5, 1); scene.add(o); return o;
  });
  // head stripes under q, K, V and y: which 4 numbers belong to which head
  [L.q, ...L.K, ...L.V, L.y].forEach((b) => [0, 1].forEach((h) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(4 * VC - 0.02, 0.012, 0.05), new THREE.MeshBasicMaterial({ color: h ? COL.h2 : COL.h1 }));
    m.position.set((h ? 2 : -2) * VC, 0, -VC * 0.75); b.group.add(m); (b.stripes = b.stripes || []).push(m);
  }));
  // row labels: l, c, r beside X, K, V
  [L.X, L.K, L.V].forEach((rows, gi) => rows.forEach((b, p) => {
    const o = b.addLabel((gi === 0 ? "x" : gi === 1 ? "k" : "v") + "_" + POS[p], -0.95, 0.12, "side"); o.center.set(1, 0.5);
  }));
  L.X[0].addLabel("x · the three tokens' vectors", 0, -0.32);
  L.K[0].addLabel("k · keys", 0, -0.32);
  L.V[0].addLabel("v · values", 0, -0.32);
  // the rows of E being read, and the row of U that wins
  const halo = () => { const m = new THREE.Mesh(new THREE.BoxGeometry(8.6 * TC, 0.5, 1.8 * TC), new THREE.MeshBasicMaterial({ color: COL.hi, transparent: true, opacity: 0.35, depthWrite: false }));
    m.position.y = 0.25; scene.add(m); return m; };
  const tag = (cy) => { const d = document.createElement("div"); d.className = "v-label small tag"; const o = new CSS2DObject(d); o.center.set(0.5, cy); scene.add(o); return o; };
  const eHalo = [0, 1, 2].map(halo), uHalo = halo(), eTags = [0, 1, 2].map(() => tag(1.15)), uTag = tag(1.15), topTags = [0, 1, 2].map(() => tag(1.1));

  // attention as arcs: the query to each key, and each value to the mix, as thick as the weight
  const arcs = new THREE.Group(); scene.add(arcs);
  function arc(a, b, lift, weight, color) {
    const mid = a.clone().add(b).multiplyScalar(0.5); mid.y = lift;
    const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(a, mid, b), 40, 0.008 + 0.05 * weight, 8, false),
                             new THREE.MeshLambertMaterial({ color, transparent: true }));
    m.userData.base = 0.25 + 0.75 * weight; m.material.opacity = m.userData.base * (cur === 3 ? 1 : 0.35);
    arcs.add(m);
  }
  function buildArcs(f) {
    arcs.children.slice().forEach((m) => { m.geometry.dispose(); m.material.dispose(); arcs.remove(m); });
    [0, 1].forEach((h) => {
      const dx = (h ? 2 : -2) * VC, col = h ? COL.h2 : COL.h1;
      [0, 1, 2].forEach((p) => {
        const w = f.attention[h][p];
        arc(new THREE.Vector3(QX + dx, 0.15, -2.6), new THREE.Vector3(QX + dx, 0.15, L.K[p].z), 1.1 + 0.35 * p, w, col);
        arc(new THREE.Vector3(QX + dx, 0.15, L.V[p].z), new THREE.Vector3(L.y.x + dx, 0.15, L.y.z), 0.9 + 0.2 * p, w, col);
      });
    });
  }
  const xNorm = () => Math.max(...view.f.x.map(maxAbs));

  // Put a window's values into the scene.
  function show(v) {
    view = v; const f = v.f;
    L.q.set(f.q); const kn = Math.max(...f.k.map(maxAbs)), vn = Math.max(...f.v.map(maxAbs));
    L.K.forEach((b, p) => b.set(f.k[p], kn)); L.V.forEach((b, p) => b.set(f.v[p], vn));
    L.A.set(f.attention[0].concat(f.attention[1]), 1); L.y.set(f.y); L.o.set(f.o);
    L.h1.set(f.h1, Math.max(maxAbs(f.h1), maxAbs(f.x[2]))); L.hid.set(f.hidden); L.h2.set(f.h2);
    const probs = Array.from(f.logits, (x, i) => v.prob(i));
    L.logit.set(probs, 1);
    L.X.forEach((b, p) => b.set(f.x[p], xNorm()));
    [0, 1, 2].forEach((p) => {
      const c = tileCenter(L.E, v.ids[p]); eHalo[p].position.set(c.x, 0.25, c.z);
      eTags[p].position.set(c.x, 0.5, c.z); eTags[p].element.textContent = POS[p] + ": " + vin[v.ids[p]];
    });
    const u = tileCenter(L.U, f.best); uHalo.position.set(u.x, 0.25, u.z); uTag.position.set(u.x, 0.5, u.z);
    uTag.element.textContent = "“" + vout[f.best] + "”";
    v.order.slice(0, 3).forEach((i, n) => {
      const p = L.logit.place(i);
      topTags[n].position.set(L.logit.x + p[0], probs[i] * L.logit.H + 0.05, L.logit.z + p[1]);
      topTags[n].element.textContent = vout[i] + " " + fmtP(v.prob(i));
      topTags[n].element.classList.toggle("win", n === 0);
    });
    buildArcs(f); blocks.forEach((b) => { b.dirty = true; });
    setChapter(cur, { force: true });
  }

  // What is shown at stage c: weights always, values once the pass has reached them, the current stage's blocks in full.
  // opts.fly: lift the tokens' rows out of E when stage 1 is entered (from stage 0 only, unless opts.fly is "always").
  function setChapter(c, o = {}) {
    const prev = cur, changed = c !== prev; cur = c;
    if (!view) return;
    const all = c === ALL;
    blocks.forEach((b) => {
      if (!b.weights) b.show = c >= b.chapter ? 1 : 0;
      const on = all || c === 0 || c === 7 && b.chapter >= 6 || b.chapter === c || c === 4 && b === L.X[2] || c === 6 && b === L.h2;
      b.setOpacity(on ? 1 : c === 7 ? 0.4 : 0.28);
      (b.labels || []).forEach((l) => { l.visible = !all && c !== 0 && (b.chapter === c || c === 7 && b.chapter === 6); });
    });
    sections.forEach((s) => { s.visible = c === 0 || all; });
    arcs.visible = c >= 3; arcs.children.forEach((m) => { m.material.opacity = m.userData.base * (c === 3 || all ? 1 : 0.35); });
    eHalo.forEach((m) => { m.visible = c <= 2 || all; });
    eTags.forEach((t) => { t.visible = c === 1; });
    uHalo.visible = uTag.visible = c >= 6;
    topTags.forEach((t) => { t.visible = c >= 6 && !(all && t !== topTags[0]); });
    const fly = c === 1 && changed && (prev === 0 || o.fly === "always");
    if (fly) liftRows();
    else if (c >= 1 && (changed || o.force)) { flyToken++; L.X.forEach((b, p) => { b.group.position.set(-8.6, 0, b.z); b.group.scale.set(1, 1, 1); b.set(view.f.x[p], xNorm()); }); }
  }

  // Each token's row of E rises out of the table and flies to its place; then the position vector is added.
  function liftRows() {
    const token = ++flyToken, f = view.f, norm = xNorm(), E = W["E.weight"];
    L.X.forEach((b, p) => {
      b.set(E[view.ids[p]], norm);
      if (reduced) return;
      const from = tileCenter(L.E, view.ids[p]), s = TC / VC, t0 = performance.now() + p * 160, to = new THREE.Vector3(-8.6, 0, b.z);
      b.group.position.copy(from); b.group.scale.set(s, s, s); b.cur.set(b.target); b.vis = 1; b.dirty = true;
      hooks.push((now) => {
        if (token !== flyToken) return false;
        const t = Math.min(1, Math.max(0, (now - t0) / 1100)), e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        b.group.position.lerpVectors(from, to, e); b.group.position.y = Math.sin(Math.PI * e) * 1.6;
        const sc = s + (1 - s) * e; b.group.scale.set(sc, sc, sc);
        return t < 1;
      });
    });
    setTimeout(() => { if (token === flyToken && cur >= 1) L.X.forEach((b, p) => b.set(f.x[p], norm)); }, reduced ? 0 : 1900);
  }

  // The camera: a tween to a stage's view (the explorer), or a view the page sets every frame (the walkthrough), which
  // the camera follows smoothly.
  let camTween = null, camAim = null;
  const fit = (v) => { const k = Math.max(1, 1.25 / camera.aspect);     // pull back on narrow stages, where less fits across
    const t = new THREE.Vector3(...v[0]); return [t, new THREE.Vector3(...v[1]).sub(t).multiplyScalar(k).add(t)]; };
  function goTo(v) {
    const [t, p] = fit(v);
    if (reduced) { target.copy(t); camera.position.copy(p); return; }
    camTween = { t0: performance.now(), fromT: target.clone(), fromP: camera.position.clone(), t, p };
  }
  function aim(a, b, s) {                                // between view a and view b, s of the way
    const [ta, pa] = fit(a), [tb, pb] = fit(b);
    camAim = [ta.lerp(tb, s), pa.lerp(pb, s)]; camTween = null;
  }

  let last = performance.now();
  (function loop(now) {
    requestAnimationFrame(loop);
    if (!visible || !view) { last = now; return; }
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (controls) target.copy(controls.target);
    if (camTween) {
      const t = Math.min(1, (now - camTween.t0) / 1300), e = 1 - Math.pow(1 - t, 3);
      target.lerpVectors(camTween.fromT, camTween.t, e); camera.position.lerpVectors(camTween.fromP, camTween.p, e);
      if (t >= 1) camTween = null;
    } else if (camAim) {
      const k = reduced ? 1 : 1 - Math.exp(-dt * 7);
      target.lerp(camAim[0], k); camera.position.lerp(camAim[1], k);
    }
    hooks = hooks.filter((h) => h(now) !== false);
    blocks.forEach((b) => b.frame(dt));
    if (controls) { controls.target.copy(target); controls.update(); } else camera.lookAt(target);
    renderer.render(scene, camera); labels.render(scene, camera);
  })(last);

  // hover: the value under the pointer
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
  labels.domElement.addEventListener("pointermove", (ev) => {
    const r = labels.domElement.getBoundingClientRect();
    ptr.set((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1);
    ray.setFromCamera(ptr, camera);
    const hit = ray.intersectObjects(hoverable.filter((m) => m.userData.block.opacity > 0.5 && m.userData.block.vis > 0.5), false)[0];
    if (!hit || hit.instanceId === undefined || !view) { tip.hidden = true; return; }
    tip.textContent = hit.object.userData.block.info(hit.instanceId);
    tip.hidden = false;
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    tip.style.left = Math.max(6, Math.min(r.width - tip.offsetWidth - 6, x + 12)) + "px"; tip.style.top = Math.max(6, y - 34) + "px";
  });
  labels.domElement.addEventListener("pointerleave", () => { tip.hidden = true; });

  function retheme() {
    theme();
    blocks.forEach((b) => { b.plate.material.color.copy(COL.plate); (b.stripes || []).forEach((m, h) => m.material.color.copy(h ? COL.h2 : COL.h1)); b.dirty = true; });
    eHalo.concat([uHalo]).forEach((m) => m.material.color.copy(COL.hi));
    if (view) buildArcs(view.f);
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);

  return { show, setChapter, goTo, aim, retheme, get chapter() { return cur; }, get view() { return view; } };
}

// The theme button every page has.
export function themeButton(onChange) {
  document.querySelector(".theme").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tct-theme", root.dataset.theme); } catch (e) {}
    onChange();
  });
}
