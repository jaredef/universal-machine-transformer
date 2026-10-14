// inside.html: one forward pass of the version 2 network (data2.js), walked through in 3D. The values drawn are the ones
// Network.prototype.forward (core.js) returns, the same function the decoder runs; the window comes live from a run of the
// sieve of Eratosthenes to 120, or is one of the three it reads that were never in training (inside-data.js).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const D = window.TCT_DATA2, TCT = window.TCT, INSIDE = window.TCT_INSIDE, W = D.weights;
const engine = new TCT.Engine(D), net = engine.net, vin = D.vin, vout = D.vout;
const $ = (id) => document.getElementById(id);
const comma = (n) => n.toLocaleString("en-US");
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const params = new URLSearchParams(location.search);

// ---------------------------------------------------------------- the window: live from the sieve, or one it never saw

const SIEVE = window.TCT_PRIMES.sieve(120), machine = TCT.parseMachine(SIEVE.start, SIEVE.rules);
const run = new TCT.Run(engine, machine, SIEVE.tape), BID = engine.id["#"];
let readStart = 1, unseenAt = -1, view = null;          // readStart: where the ring being read starts in run.T

function stepRun() {
  run.step(false);
  if (run.T[run.k + 1] === BID) readStart = run.k + 1;
}
function isHead(id) { return vin[id].indexOf(":") >= 0; }
function liveView() {
  const k = run.k, ids = run.window(), tok = (a, b) => run.tokens(a, b);
  let end = k + 2; while (end < run.n && run.T[end] !== BID) end++;
  const written = run.ringStart === null || vin[ids[1]] === "#" ? [] : tok(Math.max(run.ringStart + 1, run.n - 12), run.n);
  return { ids, live: true, pass: run.passes + 1, ring: run.utmSteps, at: k + 1 - readStart, ringLen: end - readStart,
           before: tok(Math.max(readStart, k - 12), k), after: tok(k + 3, Math.min(end, k + 15)), written,
           writtenMore: run.ringStart !== null && run.n - 12 > run.ringStart + 1 };
}
function unseenView(i) {
  const u = INSIDE.unseen[i];
  return { ids: u.window.map((t) => engine.id[t]), live: false, unseen: i, pass: u.pass + 1, ring: u.utmStep, at: u.at,
           ringLen: u.ring, before: u.before.slice(-12), after: u.after.slice(0, 12), written: null };
}
function show(v) {
  view = v;
  const f = net.forward(v.ids), lg = f.logits;
  let mx = -Infinity; for (const x of lg) mx = Math.max(mx, x);
  let z = 0; for (const x of lg) z += Math.exp(x - mx);
  const order = Array.from(lg.keys()).sort((a, b) => lg[b] - lg[a]);
  v.f = f; v.order = order; v.prob = (i) => Math.exp(lg[i] - mx) / z;
  v.rank = new Int32Array(lg.length); order.forEach((i, r) => { v.rank[i] = r; });
  v.out = D.emits[vout[f.best]]; v.exact = engine.exact(v.ids); v.trained = engine.trained.has(engine.key(v.ids));
  v.wrote = v.out.join(" ");
  drawStrip(); scene3d.window(v); chapter(cur, true);
}

// ---------------------------------------------------------------- the strip of transcript above the stage

function tokSpan(t, cls, role) {
  const s = document.createElement("span");
  s.className = "tok" + (t.indexOf(":") >= 0 ? " head" : "") + (cls ? " " + cls : "");
  s.textContent = t; if (role) s.dataset.r = role;
  return s;
}
function drawStrip() {
  const v = view, el = $("strip"); el.innerHTML = "";
  const line = (label) => { const d = document.createElement("div"); d.className = "line"; const l = document.createElement("span");
    l.className = "lab"; l.textContent = label; d.appendChild(l); el.appendChild(d); return d; };
  const r = line("reading ring " + comma(v.ring));
  if (v.before.length < v.at - 1) r.appendChild(tokSpan("…", "more"));
  v.before.forEach((t) => r.appendChild(tokSpan(t)));
  ["l", "c", "r"].forEach((role, i) => r.appendChild(tokSpan(vin[v.ids[i]], "win w-" + role, role)));
  v.after.forEach((t) => r.appendChild(tokSpan(t)));
  if (v.at + 2 + v.after.length < v.ringLen) r.appendChild(tokSpan("…", "more"));
  const w = line(vin[v.ids[1]] === "#" ? "starts ring " + comma(v.ring + 1) : "writing ring " + comma(v.ring + 1));
  if (v.written) {
    if (v.writtenMore) w.appendChild(tokSpan("…", "more"));
    v.written.forEach((t) => w.appendChild(tokSpan(t)));
  } else w.appendChild(tokSpan("…", "more"));
  const outs = v.out.length ? v.out : ["(halt)"];
  outs.forEach((t) => { const s = tokSpan(cur >= 7 ? t : "?", "out" + (cur >= 7 ? " shown" : "")); s.dataset.t = t; w.appendChild(s); });
  $("w-pass").textContent = comma(v.pass);
  $("w-pass-label").textContent = v.live ? "Forward pass" : "Forward pass, of " + comma(INSIDE.passes);
  $("w-note").textContent = v.live ? "From the sieve to 120, running live in this page." :
    "Never-seen window " + (v.unseen + 1) + " of 3, from the end of the sieve to 120.";
  const ring = (vin[v.ids[1]] === "#") ? "the ring's start" : "cell " + comma(v.at) + " of " + comma(v.ringLen - 1);
  $("w-sub").textContent = "The transcript around the three tokens the network is reading, at " + ring +
    ". It writes the token for the middle one's place in the next ring.";
}
function revealOut() {
  document.querySelectorAll("#strip .tok.out").forEach((s) => { s.textContent = cur >= 7 ? s.dataset.t : "?"; s.classList.toggle("shown", cur >= 7); });
}

// ---------------------------------------------------------------- the 3D scene

const SCENE = (function () {
  const el = $("stage"), tip = $("tip");
  el.querySelector(".v-msg").remove();
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  el.appendChild(renderer.domElement);
  const labels = new CSS2DRenderer(); labels.domElement.className = "labels3d"; el.appendChild(labels.domElement);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.1, 400);
  camera.position.set(6, 26, 30);
  const controls = new OrbitControls(camera, labels.domElement);
  controls.enableDamping = true; controls.target.set(6, 0, 0); controls.maxPolarAngle = Math.PI * 0.48;
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(-4, 10, 6); scene.add(sun);
  function resize() {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h, false); labels.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(el); resize();
  let visible = true;
  new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(el);
  return { el, tip, renderer, labels, scene, camera, controls, hooks: [], get visible() { return visible; } };
})();

const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function isDark() { return getComputedStyle(document.body).backgroundColor.match(/\d+/g).slice(0, 3).reduce((a, b) => a + +b, 0) < 300; }
const COL = { pos: new THREE.Color(), neg: new THREE.Color(), mid: new THREE.Color(), h1: new THREE.Color(), h2: new THREE.Color(),
              plate: new THREE.Color(), hi: new THREE.Color(), ink: new THREE.Color() };
function theme() {
  const dark = isDark();
  COL.pos.set(dark ? "#d95926" : "#eb6834"); COL.neg.set(dark ? "#3987e5" : "#2a78d6");
  COL.mid.set(dark ? "#4a4f59" : "#c3c2b7"); COL.h1.set(dark ? "#199e70" : "#1baf7a"); COL.h2.set(dark ? "#9a72f0" : "#8b5cf6");
  COL.plate.set(css("--rule") || "#e2e0d9"); COL.ink.set(css("--ink") || "#16181d");
  COL.hi.set(dark ? "#f2c94c" : "#e0a400");
}
theme();

const BOX = new THREE.BoxGeometry(1, 1, 1); BOX.translate(0, 0.5, 0);
const blocks = [], hoverable = [];
const tmpM = new THREE.Matrix4(), tmpC = new THREE.Color();

// A block of bars: n numbers at fixed places on the floor, each a bar whose height is the number's size.
class Bars {
  constructor(o) {
    Object.assign(this, { cell: 0.2, H: 1, chapter: 0, emph: [], animate: true, color: null }, o);
    const n = this.n;
    this.mat = new THREE.MeshLambertMaterial({ transparent: true, opacity: 1 });
    this.mesh = new THREE.InstancedMesh(BOX, this.mat, n);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.userData.block = this;
    this.group = new THREE.Group(); this.group.position.set(this.x, 0, this.z); this.group.add(this.mesh);
    SCENE.scene.add(this.group);
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
  addLabel(text, x, z, chapters, cls) {
    const d = document.createElement("div"); d.className = "v-label small" + (cls ? " " + cls : ""); d.textContent = text;
    const o = new CSS2DObject(d); o.position.set(x, 0.05, z); o.center.set(0.5, 1); this.group.add(o);
    o.userData.chapters = chapters || null; o.userData.block = this;
    (this.labels = this.labels || []).push(o);
    return o;
  }
  set(values, norm) {
    for (let i = 0; i < this.n; i++) this.target[i] = values[i];
    let m = 0; for (let i = 0; i < this.n; i++) m = Math.max(m, Math.abs(values[i]));
    this.norm = norm || m || 1; this.dirty = true;
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

// A vector of d numbers as a row along x.
function row(n, cell) { return (i) => [(i - (n - 1) / 2) * cell, 0]; }
// A matrix (rows x cols): its columns (inputs) along x, its rows (outputs) along z, or the other way round.
function grid(rows, cols, cell, transpose) {
  return (i) => { const r = Math.floor(i / cols), c = i % cols;
    return transpose ? [(r - (rows - 1) / 2) * cell, (c - (cols - 1) / 2) * cell] : [(c - (cols - 1) / 2) * cell, (r - (rows - 1) / 2) * cell]; };
}
// A lookup table: one tile of 8 bars per token, `per` tiles to a row.
const TC = 0.045, PER = 18;
function tiles(count) {
  const rows = Math.ceil(count / PER);
  return (i) => { const t = Math.floor(i / 8), j = i % 8, c = t % PER, r = Math.floor(t / PER);
    return [((c - PER / 2) * 9 + j + 0.5) * TC, (r - rows / 2) * 2 * TC]; };
}
function tileCenter(block, t) { const p = block.place(t * 8); return new THREE.Vector3(block.x + p[0] + 3.5 * TC, 0, block.z + p[1]); }
const flat = (M) => { const out = []; M.forEach((r) => r.forEach((v) => out.push(v))); return out; };
const f3 = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(3));
const POS = ["l", "c", "r"];

const VC = 0.2, WC = 0.16, d = 8, NU = W["W1.weight"].length;
const L = {};                                          // the blocks, by name
// lookup table E and positions P (chapter 1)
L.E = new Bars({ n: vin.length * 8, x: -16, z: 0, cell: TC, H: 0.35, chapter: 1, weights: true, animate: false, place: tiles(vin.length), labelFront: true,
                 label: "E · embedding table · 1,351 tokens × 8", info: (i) => "E[" + vin[Math.floor(i / 8)] + "][" + i % 8 + "] = " + f3(W["E.weight"][Math.floor(i / 8)][i % 8]) });
L.P = new Bars({ n: 24, x: -8.6, z: -2.4, cell: VC, H: 0.6, chapter: 1, weights: true, animate: false, place: grid(3, 8, VC),
                 label: "P · positions · 3 × 8", info: (i) => "P[" + POS[Math.floor(i / 8)] + "][" + i % 8 + "] = " + f3(W.P[Math.floor(i / 8)][i % 8]) });
L.X = [0, 1, 2].map((p) => new Bars({ n: 8, x: -8.6, z: -0.55 + 0.55 * p, cell: VC, H: 0.9, chapter: 1, place: row(8, VC),
                 info: (i) => "x_" + POS[p] + "[" + i + "] = " + f3(view.f.x[p][i]) }));
// queries, keys and values (chapter 2)
function weightBlock(name, key, x, z, rows, cols, chapter, label, transpose) {
  const M = W[key];
  const b = new Bars({ n: rows * cols, x, z, cell: WC, H: 0.55, chapter, weights: true, animate: false, place: grid(rows, cols, WC, transpose), label,
    info: (i) => name + "[" + Math.floor(i / cols) + "][" + i % cols + "] = " + f3(M[Math.floor(i / cols)][i % cols]) });
  b.set(flat(M)); return b;
}
L.Wq = weightBlock("Wq", "Wq.weight", -5.2, -2.6, 8, 8, 2, "Wq · 8 × 8");
L.Wk = weightBlock("Wk", "Wk.weight", -5.2, 0, 8, 8, 2, "Wk · 8 × 8");
L.Wv = weightBlock("Wv", "Wv.weight", -5.2, 2.6, 8, 8, 2, "Wv · 8 × 8");
const QX = -2.4;
L.q = new Bars({ n: 8, x: QX, z: -2.6, cell: VC, H: 0.9, chapter: 2, place: row(8, VC), label: "q · r's query", info: (i) => "q[" + i + "] = " + f3(view.f.q[i]) + " (head " + (i < 4 ? 1 : 2) + ")" });
L.K = [0, 1, 2].map((p) => new Bars({ n: 8, x: QX, z: -0.55 + 0.55 * p, cell: VC, H: 0.9, chapter: 2, place: row(8, VC),
  info: (i) => "k_" + POS[p] + "[" + i + "] = " + f3(view.f.k[p][i]) + " (head " + (i < 4 ? 1 : 2) + ")" }));
L.V = [0, 1, 2].map((p) => new Bars({ n: 8, x: QX, z: 2.05 + 0.55 * p, cell: VC, H: 0.9, chapter: 2, place: row(8, VC),
  info: (i) => "v_" + POS[p] + "[" + i + "] = " + f3(view.f.v[p][i]) + " (head " + (i < 4 ? 1 : 2) + ")" }));
// attention (chapter 3)
L.A = new Bars({ n: 6, x: 0.6, z: 0, cell: 0.3, H: 1.3, chapter: 3, place: (i) => [((i % 3) - 1) * 0.34, (Math.floor(i / 3) - 0.5) * 0.5],
  label: "attention weights", info: (i) => "head " + (Math.floor(i / 3) + 1) + " on " + POS[i % 3] + ": " + view.f.attention[Math.floor(i / 3)][i % 3].toFixed(3),
  color: (i, v, out) => out.copy(COL.mid).lerp(i < 3 ? COL.h1 : COL.h2, 0.25 + 0.75 * v) });
L.y = new Bars({ n: 8, x: 0.6, z: 2.6, cell: VC, H: 0.9, chapter: 3, place: row(8, VC), label: "y · the heads' mix of values", info: (i) => "y[" + i + "] = " + f3(view.f.y[i]) });
// back into the stream (chapter 4)
L.Wo = weightBlock("Wo", "Wo.weight", 3, 2.6, 8, 8, 4, "Wo · 8 × 8");
L.o = new Bars({ n: 8, x: 5.3, z: 2.6, cell: VC, H: 0.9, chapter: 4, place: row(8, VC), label: "o · attention's update", info: (i) => "o[" + i + "] = " + f3(view.f.o[i]) });
L.h1 = new Bars({ n: 8, x: 5.3, z: 0, cell: VC, H: 0.9, chapter: 4, place: row(8, VC), label: "h = x_r + o", info: (i) => "h[" + i + "] = " + f3(view.f.h1[i]) });
// the MLP (chapter 5): W1's inputs along x, its 32 units along z; W2 the other way round
// the second row of the scene, read like a second line of text: the MLP, the unembedding, the probabilities
const R2 = 10;
L.W1 = weightBlock("W1", "W1.weight", -17.2, R2, NU, 8, 5, "W1 · 32 × 8");
L.hid = new Bars({ n: NU, x: -15.95, z: R2, cell: WC, H: 0.9, chapter: 5, place: (i) => [0, (i - (NU - 1) / 2) * WC], label: "ReLU",
  info: (i) => "unit " + i + ": " + f3(view.f.pre[i]) + (view.f.pre[i] > 0 ? ", on" : ", off (ReLU makes it 0)") });
L.W2 = weightBlock("W2", "W2.weight", -14.6, R2, 8, NU, 5, "W2 · 8 × 32", true);
L.h2 = new Bars({ n: 8, x: -14.6, z: R2 + 3.2, cell: VC, H: 0.9, chapter: 5, place: row(8, VC), label: "h + the MLP's update", info: (i) => "h[" + i + "] = " + f3(view.f.h2[i]) });
// the unembedding and the scores (chapter 6)
L.U = new Bars({ n: vout.length * 8, x: -6, z: R2, labelFront: true, cell: TC, H: 0.35, chapter: 6, weights: true, animate: false, place: tiles(vout.length),
                 label: "U · unembedding · 1,356 outputs × 8", info: (i) => "U[" + vout[Math.floor(i / 8)] + "][" + i % 8 + "] = " + f3(W["U.weight"][Math.floor(i / 8)][i % 8]) });
const LG = 37, LC = 0.14;
L.logit = new Bars({ n: vout.length, x: 3.2, z: R2, cell: LC, H: 1.8, chapter: 6, labelFront: true, place: (i) => [((i % LG) - (LG - 1) / 2) * LC, (Math.floor(i / LG) - (LG - 1) / 2) * LC],
  label: "a probability for each of the 1,356 outputs", info: (i) => "“" + vout[i] + "”: score " + f3(view.f.logits[i]) + ", probability " + fmtP(view.prob(i)),
  color: (i, v, out) => i === view.f.best ? out.copy(COL.pos) : view.rank[i] < 3 ? out.copy(COL.mid).lerp(COL.pos, 0.45) : out.copy(COL.mid) });
L.E.set(flat(W["E.weight"])); L.U.set(flat(W["U.weight"])); L.P.set(flat(W.P));
function fmtP(p) { return p > 0.9995 ? "1.000" : p < 0.0005 ? (p < 1e-6 ? "< 0.000001" : p.toFixed(6)) : p.toFixed(3); }

// what the parts are, for the overview
const sections = [["The tokens' rows", -16, -3.7], ["Attention", -3, -3.7], ["The MLP", -15.9, R2 - 3], ["The unembedding", -6, R2 - 3.7],
  ["The probabilities", 3.2, R2 - 3]].map(([t, x, z]) => {
  const d = document.createElement("div"); d.className = "v-label sec"; d.textContent = t;
  const o = new CSS2DObject(d); o.position.set(x, 0, z); o.center.set(0.5, 1); SCENE.scene.add(o); return o;
});

// head stripes under q, K, V and y: which 4 numbers belong to which head
function stripes(b) {
  [0, 1].forEach((h) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(4 * VC - 0.02, 0.012, 0.05), new THREE.MeshBasicMaterial({ color: h ? COL.h2 : COL.h1 }));
    m.position.set((h ? 2 : -2) * VC, 0, -VC * 0.75); b.group.add(m); (b.stripes = b.stripes || []).push(m);
  });
}
[L.q, ...L.K, ...L.V, L.y].forEach(stripes);

// row labels: l, c, r beside X, K, V
[L.X, L.K, L.V].forEach((rows, gi) => rows.forEach((b, p) => {
  const t = (gi === 0 ? "x" : gi === 1 ? "k" : "v") + "_" + POS[p];
  const o = b.addLabel(t, -0.95, 0.12, null, "side"); o.center.set(1, 0.5);
}));
L.X[0].addLabel("x · the three tokens' vectors", 0, -0.32);
L.K[0].addLabel("k · keys", 0, -0.32);
L.V[0].addLabel("v · values", 0, -0.32);
// the selected rows of E and U
const halo = (color) => { const m = new THREE.Mesh(new THREE.BoxGeometry(8.6 * TC, 0.5, 1.8 * TC), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.35, depthWrite: false }));
  m.position.y = 0.25; SCENE.scene.add(m); return m; };
const eHalo = [0, 1, 2].map(() => halo(COL.hi)), uHalo = halo(COL.hi);
const eTags = [0, 1, 2].map((p) => { const d = document.createElement("div"); d.className = "v-label small tag"; const o = new CSS2DObject(d); o.center.set(0.5, 1.15); SCENE.scene.add(o); return o; });
const uTag = (() => { const d = document.createElement("div"); d.className = "v-label small tag"; const o = new CSS2DObject(d); o.center.set(0.5, 1.15); SCENE.scene.add(o); return o; })();
const topTags = [0, 1, 2].map(() => { const d = document.createElement("div"); d.className = "v-label small tag"; const o = new CSS2DObject(d); o.center.set(0.5, 1.1); SCENE.scene.add(o); return o; });

// attention as arcs: query to each key, and each value to the mix, as thick as the weight
const arcs = new THREE.Group(); SCENE.scene.add(arcs);
function arc(a, b, lift, weight, color) {
  const mid = a.clone().add(b).multiplyScalar(0.5); mid.y = lift;
  const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
  const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 40, 0.008 + 0.05 * weight, 8, false),
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

// ---------------------------------------------------------------- the stages of the pass

const CH = [
  { title: "The window", view: [[-7.4, 0, 5.4], [-7.4, 23.5, 27]] },
  { title: "Look up the tokens", view: [[-13.4, 0, 0.3], [-13.4, 10.5, 11.6]] },
  { title: "Queries, keys and values", view: [[-3.8, 0, 0.3], [-3.8, 7, 7.4]] },
  { title: "Attention", view: [[-1, 0, 0.6], [-0.4, 6.4, 6.9]] },
  { title: "Back into the stream", view: [[3.6, 0, 1.4], [3.6, 6, 7]] },
  { title: "The MLP", view: [[-15.8, 0, R2 + 0.6], [-15.8, 7.2, R2 + 7.6]] },
  { title: "Scoring every output", view: [[-1.6, 0, R2], [-1.6, 10, R2 + 10.5]] },
  { title: "What it writes", view: [[3.2, 0.7, R2], [3.2, 3.6, R2 + 7.6]] },
];
let cur = Math.max(0, Math.min(CH.length - 1, +params.get("ch") || 0)), flyToken = 0;

function chapter(c, sameView) {
  const changed = c !== cur; cur = c;
  document.querySelectorAll("#chapters button").forEach((b, i) => { b.classList.toggle("on", i === c); b.setAttribute("aria-current", i === c ? "step" : "false"); });
  $("ch-num").textContent = (c + 1) + " of " + CH.length;
  $("ch-title").textContent = CH[c].title;
  $("ch-text").innerHTML = TEXT[c](view);
  $("prev").disabled = c === 0; $("next").disabled = c === CH.length - 1;
  revealOut();
  const v = view, f = v.f;
  // what is shown: weights always; values once the pass has reached them
  blocks.forEach((b) => {
    if (!b.weights) b.show = c >= b.chapter ? 1 : 0;
    const on = c === 0 || c === 7 && b.chapter >= 6 || b.chapter === c || c === 4 && b === L.X[2] || c === 6 && b === L.h2;
    b.setOpacity(on ? 1 : c === 7 ? 0.4 : 0.28);
    (b.labels || []).forEach((o) => { o.visible = c !== 0 && (b.chapter === c || c === 7 && b.chapter === 6); });
  });
  sections.forEach((o) => { o.visible = c === 0; });
  arcs.visible = c >= 3; arcs.children.forEach((m) => { m.material.opacity = m.userData.base * (c === 3 ? 1 : 0.35); });
  eHalo.forEach((m, p) => { m.visible = c >= 1 && c <= 2 || c === 0; });
  eTags.forEach((o) => { o.visible = c === 1; });
  uHalo.visible = c >= 6; uTag.visible = c >= 6;
  topTags.forEach((o) => { o.visible = c >= 6; });
  if (c === 1 && changed) fly(); else if (c >= 1) L.X.forEach((b, p) => { b.group.position.set(-8.6, 0, b.z); b.group.scale.set(1, 1, 1); b.set(f.x[p], xNorm()); });
  if (!sameView || changed) moveCamera(CH[c].view);
}
function xNorm() { let m = 0; view.f.x.forEach((v) => v.forEach((x) => { m = Math.max(m, Math.abs(x)); })); return m; }

// Each token's row of E flies out of the table to its place, and the position vector is added.
function fly() {
  const token = ++flyToken, f = view.f, norm = xNorm(), E = W["E.weight"];
  L.X.forEach((b, p) => {
    b.set(E[view.ids[p]], norm);
    if (reduced) return;
    const from = tileCenter(L.E, view.ids[p]), s = TC / VC, t0 = performance.now() + p * 160;
    b.group.position.copy(from); b.group.scale.set(s, s, s); b.cur.set(b.target); b.vis = 1; b.dirty = true;
    const to = new THREE.Vector3(-8.6, 0, b.z);
    SCENE.hooks.push(function step(now) {
      if (token !== flyToken) return false;
      const t = Math.min(1, Math.max(0, (now - t0) / 1100)), e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      b.group.position.lerpVectors(from, to, e); b.group.position.y = Math.sin(Math.PI * e) * 1.6;
      const sc = s + (1 - s) * e; b.group.scale.set(sc, sc, sc);
      return t < 1;
    });
  });
  setTimeout(() => { if (token === flyToken && cur >= 1) L.X.forEach((b, p) => b.set(f.x[p], norm)); }, reduced ? 0 : 1900);
}

let camTween = null;
function moveCamera(v) {
  const k = Math.max(1, 1.25 / SCENE.camera.aspect);     // pull back on narrow stages, where less fits across
  const target = new THREE.Vector3(...v[0]), pos = new THREE.Vector3(...v[1]).sub(target).multiplyScalar(k).add(target);
  if (reduced) { SCENE.controls.target.copy(target); SCENE.camera.position.copy(pos); return; }
  camTween = { t0: performance.now(), fromT: SCENE.controls.target.clone(), fromP: SCENE.camera.position.clone(), target, pos };
}
SCENE.controls.addEventListener("start", () => { camTween = null; });

// Put this window's values into the scene.
const scene3d = {
  window(v) {
    const f = v.f;
    L.q.set(f.q); const kn = Math.max(...f.k.flatMap((r) => Array.from(r, Math.abs))), vn = Math.max(...f.v.flatMap((r) => Array.from(r, Math.abs)));
    L.K.forEach((b, p) => b.set(f.k[p], kn)); L.V.forEach((b, p) => b.set(f.v[p], vn));
    L.A.set(f.attention[0].concat(f.attention[1]), 1); L.y.set(f.y); L.o.set(f.o);
    const hn = Math.max(...Array.from(f.h1, Math.abs), ...Array.from(f.x[2], Math.abs));
    L.h1.set(f.h1, hn); L.hid.set(f.hidden); L.h2.set(f.h2);
    const lg = f.logits, probs = Array.from(lg, (x, i) => v.prob(i));
    L.logit.set(probs, 1);
    L.X.forEach((b, p) => b.set(f.x[p], xNorm()));
    [0, 1, 2].forEach((p) => {
      const c = tileCenter(L.E, v.ids[p]); eHalo[p].position.set(c.x, 0.25, c.z);
      eTags[p].position.set(c.x, 0.5, c.z); eTags[p].element.textContent = POS[p] + ": " + vin[v.ids[p]];
    });
    const u = tileCenter(L.U, f.best); uHalo.position.set(u.x, 0.25, u.z); uTag.position.set(u.x, 0.5, u.z);
    uTag.element.textContent = "“" + vout[f.best] + "”";
    v.order.slice(0, 3).forEach((i, n) => {
      const p = L.logit.place(i), h = probs[i] * L.logit.H;
      topTags[n].position.set(L.logit.x + p[0], h + 0.05, L.logit.z + p[1]);
      topTags[n].element.textContent = vout[i] + " " + fmtP(v.prob(i));
      topTags[n].element.classList.toggle("win", n === 0);
    });
    buildArcs(f); blocks.forEach((b) => { b.dirty = true; });
  },
};

// ---------------------------------------------------------------- the words for each stage

const code = (t) => "<code>" + t.replace(/&/g, "&amp;").replace(/</g, "&lt;") + "</code>";
function describe(t) {
  if (t === "#") return "the ring's start";
  const i = t.indexOf(":");
  if (i < 0) return "a cell holding " + code(t);
  const st = t.slice(0, i);
  return "the universal machine's head, in state " + code(st) + " on " + code(t.slice(i + 1)) + " (" + TCT.phase(st) + ")";
}
function pct(x) { return Math.round(x * 100) + "%"; }
const TEXT = [
  (v) => "<p>The network reads three tokens of the transcript:</p><ul class='toks'>" + [0, 1, 2].map((p) =>
    "<li><b>" + POS[p] + "</b> " + code(vin[v.ids[p]]) + ", " + describe(vin[v.ids[p]]) + "</li>").join("") +
    "</ul><p>Its whole job is the token that goes at <b>c</b>'s place in the next ring. From left to right below: the " +
    "embedding table, attention, the MLP, and the unembedding, with the numbers this pass computes between them.</p>" +
    "<p class='muted'>" + (v.trained ? "This window was in its training data." : "It never saw this window in training.") + "</p>",
  (v) => "<p>Each token has a row of 8 numbers in the embedding table <b>E</b>: 1,351 tokens, 10,808 parameters, " +
    "45% of the network. The three rows are copied out (highlighted in the table).</p><p>Then a position vector from <b>P</b> " +
    "is added to each, so the same token reads differently as l, c, or r. These three vectors of 8 numbers are all the " +
    "network knows about the window.</p>",
  (v) => "<p>Three 8 × 8 matrices. Only r's position asks a question, its <b>query</b> q = Wq·x<sub>r</sub>, the way a " +
    "language model predicts from its last token. Every position answers with a <b>key</b>, k = Wk·x, and offers a " +
    "<b>value</b>, v = Wv·x.</p><p>The 8 numbers are split between two attention heads of 4 each, " +
    "<span class='hk h1'>head 1</span> and <span class='hk h2'>head 2</span>.</p>",
  (v) => {
    const a = v.f.attention, line = (h) => "<span class='hk h" + (h + 1) + "'>Head " + (h + 1) + "</span> puts " +
      [0, 1, 2].map((p) => "<b>" + pct(a[h][p]) + "</b> on " + POS[p]).join(", ") + ".";
    return "<p>Each head compares the query with the three keys (a dot product, divided by 2) and turns the scores into weights " +
      "that add up to 1. The arcs are as thick as the weights.</p><p>" + line(0) + "<br>" + line(1) + "</p><p>Each head then takes " +
      "that mix of the values; together they make <b>y</b>.</p>";
  },
  (v) => {
    const n = (x) => Math.max(...Array.from(x, Math.abs));
    return "<p><b>Wo</b> turns the mix into an update, <b>o</b>, which is added to r's own vector: h = x<sub>r</sub> + o. " +
      "That running sum is the <b>residual stream</b>, everything the pass knows so far.</p>" +
      (n(v.f.o) > n(v.f.x[2]) ? "<p>The update is larger than what it is added to (its largest number is " + f3(n(v.f.o)) +
      ", against " + f3(n(v.f.x[2])) + "): attention has rewritten r's vector with what it found at l and c.</p>" :
      "<p>The update's largest number is " + f3(n(v.f.o)) + ", against " + f3(n(v.f.x[2])) + " in r's own vector.</p>");
  },
  (v) => {
    const on = Array.from(v.f.hidden).filter((x) => x > 0).length, n = (x) => Math.max(...Array.from(x, Math.abs));
    return "<p>32 units. Each takes a weighted sum of the stream (its row of <b>W1</b>) plus a bias, and passes it on only if " +
      "it is positive (ReLU). For this window <b>" + on + " of 32</b> are on.</p><p><b>W2</b> adds what they found back into " +
      "the stream; its largest number goes from " + f3(n(v.f.h1)) + " to " + f3(n(v.f.h2)) + ".</p><p class='muted'>Between the two tables, attention and the MLP are 832 parameters.</p>";
  },
  (v) => "<p><b>U</b> has a row for every output the network can write: 1,351 single tokens, 4 pairs (written when the " +
    "machine's head moves past the end of the ring), and writing nothing (the machine halts). An output's score is its " +
    "row's dot product with the stream, plus a bias, and a softmax turns the scores into probabilities. The bars are the " +
    "probabilities.</p><p>The highest:</p><ol class='toks'>" + v.order.slice(0, 3).map((i) =>
    "<li>" + code(vout[i]) + " <span class='muted'>score " + f3(v.f.logits[i]) + ",</span> probability " + fmtP(v.prob(i)) + "</li>").join("") +
    "</ol>",
  (v) => {
    const right = v.wrote === v.exact, w = v.wrote ? code(v.wrote) : "nothing (it halts)";
    let s = "<p>It writes " + w + ". The universal machine's exact rule says " + (v.exact ? code(v.exact) : "halt") + ": " +
      (right ? "<b class='ok'>the same ✓</b>" : "<b class='bad'>different ✗</b>") + ".</p>";
    if (v.live) s += "<p>" + (v.trained ? "This window was in its training data. " : "It never saw this window in training. ") +
      "Then the window moves one token to the right, and the next pass begins. Running the whole sieve to 120 takes " +
      comma(INSIDE.passes) + " of these.</p>";
    else s += "<p>It never saw this window in training, and gives the right answer probability " + fmtP(v.prob(v.f.best)) + ". The sieve to 120 reads 12,406 different windows; only 3 were not in " +
      "training, and all three come in the last 0.2% of its " + comma(INSIDE.passes) + " passes, here, as the universal machine " +
      "jumps to the next state by its binary address. It gets all three right.</p>";
    return s;
  },
];

// ---------------------------------------------------------------- the loop, hover, and controls

let last = performance.now();
(function loop(now) {
  requestAnimationFrame(loop);
  if (!SCENE.visible || !view) { last = now; return; }
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (camTween) {
    const t = Math.min(1, (now - camTween.t0) / 1300), e = 1 - Math.pow(1 - t, 3);
    SCENE.controls.target.lerpVectors(camTween.fromT, camTween.target, e);
    SCENE.camera.position.lerpVectors(camTween.fromP, camTween.pos, e);
    if (t >= 1) camTween = null;
  }
  SCENE.hooks = SCENE.hooks.filter((h) => h(now) !== false);
  blocks.forEach((b) => b.frame(dt));
  SCENE.controls.update();
  SCENE.renderer.render(SCENE.scene, SCENE.camera); SCENE.labels.render(SCENE.scene, SCENE.camera);
})(last);

const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
SCENE.labels.domElement.addEventListener("pointermove", (ev) => {
  const r = SCENE.labels.domElement.getBoundingClientRect();
  ptr.set((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1);
  ray.setFromCamera(ptr, SCENE.camera);
  const hit = ray.intersectObjects(hoverable.filter((m) => m.userData.block.opacity > 0.5 && m.userData.block.vis > 0.5), false)[0];
  const tip = SCENE.tip;
  if (!hit || hit.instanceId === undefined || !view) { tip.hidden = true; return; }
  tip.textContent = hit.object.userData.block.info(hit.instanceId);
  tip.hidden = false;
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  tip.style.left = Math.min(r.width - tip.offsetWidth - 6, x + 12) + "px"; tip.style.top = Math.max(6, y - 34) + "px";
});
SCENE.labels.domElement.addEventListener("pointerleave", () => { SCENE.tip.hidden = true; });

const rail = $("chapters");
CH.forEach((c, i) => {
  const li = document.createElement("li"), b = document.createElement("button");
  b.type = "button"; b.textContent = i + 1; b.title = c.title; b.setAttribute("aria-label", (i + 1) + ": " + c.title);
  b.addEventListener("click", () => { stopTour(); chapter(i); });
  li.appendChild(b); rail.appendChild(li);
});
$("prev").addEventListener("click", () => { stopTour(); if (cur > 0) chapter(cur - 1); });
$("next").addEventListener("click", () => { stopTour(); if (cur < CH.length - 1) chapter(cur + 1); });
function backToLive() { if (!view.live) unseenAt = -1; }
$("next-pass").addEventListener("click", () => { stopTour(); if (view.live) stepRun(); backToLive(); show(liveView()); });
function nextHead() {
  let n = 0;
  do { stepRun(); n++; } while (!run.window().some(isHead) && n < 200000 && !run.verdict);
}
$("next-head").addEventListener("click", () => { stopTour(); if (view.live) nextHead(); else if (!run.window().some(isHead)) nextHead(); backToLive(); show(liveView()); });
$("unseen").addEventListener("click", () => { stopTour(); unseenAt = (unseenAt + 1) % INSIDE.unseen.length; show(unseenView(unseenAt)); });

let tour = null;
function stopTour() { if (tour) { clearInterval(tour); tour = null; $("tour").textContent = "▶ Walk through"; } }
$("tour").addEventListener("click", () => {
  if (tour) return stopTour();
  $("tour").textContent = "❚❚ Pause";
  if (cur === CH.length - 1) chapter(0);
  tour = setInterval(() => {
    if (cur < CH.length - 1) chapter(cur + 1);
    else { if (view.live) nextHead(); show(view.live ? liveView() : unseenView(unseenAt)); chapter(0); }
  }, reduced ? 6000 : 5200);
});

document.querySelector(".theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("tct-theme", root.dataset.theme); } catch (e) {}
  retheme();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
function retheme() {
  theme();
  blocks.forEach((b) => { b.plate.material.color.copy(COL.plate); (b.stripes || []).forEach((m, h) => m.material.color.copy(h ? COL.h2 : COL.h1)); b.dirty = true; });
  eHalo.concat([uHalo]).forEach((m) => m.material.color.copy(COL.hi));
  if (view) buildArcs(view.f);
}

// ---------------------------------------------------------------- the parameter budget

(function budget() {
  const n = (k) => flat(Array.isArray(W[k][0]) ? W[k] : [W[k]]).length;
  const rows = [
    ["E", "embedding table", vin.length + " × 8", n("E.weight")],
    ["P", "positions", "3 × 8", n("P")],
    ["Wq, Wk, Wv, Wo", "attention", "4 × (8 × 8)", n("Wq.weight") + n("Wk.weight") + n("Wv.weight") + n("Wo.weight")],
    ["W1, W2 and biases", "the MLP", "8 → 32 → 8", n("W1.weight") + n("W1.bias") + n("W2.weight") + n("W2.bias")],
    ["U and bias", "unembedding", vout.length + " × 8 + " + vout.length, n("U.weight") + n("U.bias")],
  ];
  const total = rows.reduce((s, r) => s + r[3], 0);
  $("budget").innerHTML = "<thead><tr><th>Block</th><th>What it is</th><th>Shape</th><th class='n'>Parameters</th><th class='n'>Share</th></tr></thead><tbody>" +
    rows.map((r) => "<tr><td>" + r[0] + "</td><td>" + r[1] + "</td><td class='mono'>" + r[2] + "</td><td class='n mono'>" + comma(r[3]) +
      "</td><td class='n mono'>" + (r[3] / total * 100).toFixed(1) + "%</td></tr>").join("") +
    "</tbody><tfoot><tr><td colspan='3'>Total</td><td class='n mono'>" + comma(total) + "</td><td class='n mono'>100%</td></tr></tfoot>";
  if (total !== D.params) console.warn("parameter count", total, "differs from", D.params);
})();

// ---------------------------------------------------------------- start: a pass at the head, two rings in

const w0 = params.get("w");
if (w0 && /^unseen[123]$/.test(w0)) { unseenAt = +w0.slice(6) - 1; show(unseenView(unseenAt)); }
else {
  while (run.utmSteps < 2 || !isHead(run.window()[1])) stepRun();
  show(liveView());
}
chapter(cur);
window.__inside = { view: () => view, chapter: (c) => chapter(c), forward: (ids) => net.forward(ids) };
