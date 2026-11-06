// umt.html: the Universal Machine Transformer page. Every 3D view draws into its own slot of one shared WebGL canvas
// (a page can only hold a few WebGL contexts). The numbers and weights come from the code and the trained network.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { engine, vin, vout, D, W, TCT } from "./inside-scene.js";

const $ = (id) => document.getElementById(id);
const comma = (n) => n.toLocaleString("en-US");
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

// ---------------------------------------------------------------- palette (the chart room)
const C = (h) => new THREE.Color(h);
const NAVY = C("#0B1F33"), MID = C("#D8DEE2"), PALE = C("#B0BEC8"), ORANGE = C("#F05A1A"), GLASS = C("#3E9C8C"),
      FOG = C("#E2E7EA"), SLATE = C("#5A6A78"), WHITE = C("#FFFFFF");
const tmpC = new THREE.Color(), tmpM = new THREE.Matrix4();
// A weight's colour: deep navy for positive, pale slate for negative, fog near zero.
function wColor(v, norm, out = new THREE.Color()) {
  const t = Math.min(1, Math.abs(v) / norm);
  return v >= 0 ? out.copy(MID).lerp(NAVY, 0.15 + 0.85 * t) : out.copy(MID).lerp(PALE, 0.3 + 0.7 * t);
}
function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

// ---------------------------------------------------------------- one canvas, many views
const canvas = $("gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
const views = [];
const DIR = new THREE.Vector3(1, 0.82, 1).normalize();          // the isometric view, as in the design
function addView(id, opts = {}) {
  const el = $(id);
  const scene = new THREE.Scene();
  // tops at their true colour, the two visible sides at about 0.86 and 0.75 of it, as in the design's drawings
  scene.add(new THREE.AmbientLight(0xffffff, 0.65 * Math.PI));
  const sun = new THREE.DirectionalLight(0xffffff, 0.42 * Math.PI); sun.position.set(0.3, 1, 0.6); scene.add(sun);
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -2000, 2000);
  const root = new THREE.Group(); scene.add(root);
  const v = { el, scene, camera, root, pad: opts.pad || 1.06, ext: null, aspect: 0, tick: null, controls: null, hover: null, onVisible: null, seen: false };
  views.push(v);
  return v;
}
// Frame the view's content: the camera looks along DIR at the centre, and the frustum fits the content's outline.
// The points that outline the content: each bar's foot corners and top, and the corners of everything else.
function outline(root) {
  root.updateMatrixWorld(true);
  const pts = [], m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
  root.traverse((o) => {
    if (!o.visible) return;
    if (o.isInstancedMesh) {
      for (let i = 0; i < o.count; i++) {
        o.getMatrixAt(i, m); m.premultiply(o.matrixWorld); m.decompose(p, q, sc);
        const hw = sc.x / 2, hd = sc.z / 2;
        pts.push(p.clone().add(new THREE.Vector3(-hw, 0, -hd)), p.clone().add(new THREE.Vector3(hw, 0, hd)), p.clone().add(new THREE.Vector3(hw, 0, -hd)),
                 p.clone().add(new THREE.Vector3(-hw, 0, hd)), p.clone().add(new THREE.Vector3(0, sc.y, 0)));
      }
    } else if ((o.isMesh || o.isLineSegments) && o.geometry && o.geometry.attributes.position) {
      const pos = o.geometry.attributes.position, step = Math.max(1, Math.floor(pos.count / 2000));   // the vertices themselves:
      for (let i = 0; i < pos.count; i += step) pts.push(new THREE.Vector3().fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld));  // a box would overreach
    }
  });
  return pts;
}
function fit(v) {
  const pts = outline(v.root), box = new THREE.Box3().setFromPoints(pts), c = box.getCenter(new THREE.Vector3());
  v.camera.position.copy(c).addScaledVector(DIR, 500); v.camera.up.set(0, 1, 0); v.camera.lookAt(c); v.camera.updateMatrixWorld();
  const inv = v.camera.matrixWorldInverse, p = new THREE.Vector3();
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const w of pts) {
    p.copy(w).applyMatrix4(inv);
    x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
  }
  v.ext = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: (x1 - x0) / 2, h: (y1 - y0) / 2 }; v.aspect = 0;
  v.center = c;
  if (v.el.classList.contains("orbit") && !v.controls) {
    v.controls = new OrbitControls(v.camera, v.el);
    v.controls.enableZoom = false; v.controls.enablePan = false; v.controls.enableDamping = true;
  }
  if (v.controls) { v.controls.target.copy(c); v.controls.update(); }
}
function frustum(v, aspect) {
  if (Math.abs(aspect - v.aspect) < 1e-3) return;
  v.aspect = aspect;
  let w = v.ext.w * v.pad, h = v.ext.h * v.pad;
  if (w / h > aspect) h = w / aspect; else w = h * aspect;
  Object.assign(v.camera, { left: v.ext.cx - w, right: v.ext.cx + w, top: v.ext.cy + h, bottom: v.ext.cy - h });
  v.camera.updateProjectionMatrix();
}
let last = performance.now();
(function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  const Wd = canvas.clientWidth, Hd = canvas.clientHeight;
  if (canvas.width !== Math.floor(Wd * renderer.getPixelRatio()) || canvas.height !== Math.floor(Hd * renderer.getPixelRatio())) renderer.setSize(Wd, Hd, false);
  renderer.setScissorTest(false); renderer.clear(); renderer.setScissorTest(true);
  for (const v of views) {
    const r = v.el.getBoundingClientRect();
    if (r.bottom < 0 || r.top > Hd || r.width < 2 || r.height < 2) continue;
    if (!v.seen) { v.seen = true; if (v.onVisible) v.onVisible(now); }
    if (v.tick) v.tick(now / 1000, dt);
    if (v.controls) v.controls.update();
    frustum(v, r.width / r.height);
    const y = Hd - r.bottom;
    renderer.setViewport(r.left, y, r.width, r.height); renderer.setScissor(r.left, y, r.width, r.height);
    renderer.render(v.scene, v.camera);
  }
})(last);

// hover: the value under the pointer, for views that set v.hover(mesh, instanceId)
const tip = $("tip"), ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
function hoverable(v, meshes, text) {
  v.el.addEventListener("pointermove", (e) => {
    const r = v.el.getBoundingClientRect();
    ptr.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
    ray.setFromCamera(ptr, v.camera);
    const hit = ray.intersectObjects(meshes(), false)[0];
    if (!hit || hit.instanceId === undefined) { tip.hidden = true; return; }
    const s = text(hit.object, hit.instanceId);
    if (!s) { tip.hidden = true; return; }
    tip.textContent = s; tip.hidden = false;
    tip.style.left = Math.min(innerWidth - tip.offsetWidth - 8, e.clientX + 14) + "px"; tip.style.top = (e.clientY - tip.offsetHeight - 10) + "px";
  });
  v.el.addEventListener("pointerleave", () => { tip.hidden = true; });
}

// ---------------------------------------------------------------- bars
const BOX = new THREE.BoxGeometry(1, 1, 1); BOX.translate(0, 0.5, 0);
function barMesh(n) {
  const m = new THREE.InstancedMesh(BOX, new THREE.MeshLambertMaterial({ color: 0xffffff }), n);
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < n; i++) m.setColorAt(i, WHITE);
  return m;
}
function setBar(m, i, x, z, h, w, col) {
  tmpM.makeScale(w, Math.max(0.002, h), w); tmpM.setPosition(x, 0, z);
  m.setMatrixAt(i, tmpM); m.setColorAt(i, col);
}
function done(m) { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; m.computeBoundingSphere(); }
function plate(x0, z0, x1, z1, y, color, opacity = 1) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0, 0.04, z1 - z0),
                           new THREE.MeshLambertMaterial({ color, transparent: opacity < 1, opacity }));
  m.position.set((x0 + x1) / 2, y - 0.02, (z0 + z1) / 2); return m;
}
function tube(points, radius, color, opacity = 1) {
  const curve = new THREE.CatmullRomCurve3(points);
  return new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(16, points.length * 8), radius, 8, false),
                        new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity }));
}
function arcPoints(a, b, lift, n = 24) {
  const out = [];
  for (let i = 0; i <= n; i++) { const t = i / n; const p = a.clone().lerp(b, t); p.y += Math.sin(Math.PI * t) * lift; out.push(p); }
  return out;
}

// ---------------------------------------------------------------- the network, as the trainer lays it out
// One flat array of 23,844 numbers in the trainer's layout (train-core.js), so the shipped weights, a fresh random start,
// and a network training live on the GPU are all drawn the same way.
const TC = window.TrainCore, LAY = TC.layout(vin.length, vout.length, 8);
const TENSORS = [["E", "E.weight"], ["P", "P"], ["Wq", "Wq.weight"], ["Wk", "Wk.weight"], ["Wv", "Wv.weight"], ["Wo", "Wo.weight"],
  ["W1", "W1.weight"], ["b1", "W1.bias"], ["W2", "W2.weight"], ["b2", "W2.bias"], ["U", "U.weight"], ["bU", "U.bias"]];
const SHIPPED = (() => { const p = new Float32Array(LAY.size); TENSORS.forEach(([n, k]) => { p.set(W[k].flat(), LAY.off[n].at); }); return p; })();
const INIT = TC.init(LAY, 0);                               // the trainer's own random start, seed 0
function toWeights(p) {                                     // the flat array back into core.js's nested weights
  const w = {};
  TENSORS.forEach(([n, k]) => { const o = LAY.off[n], rows = []; for (let r = 0; r < o.rows; r++) rows.push(Array.from(p.subarray(o.at + r * o.cols, o.at + (r + 1) * o.cols))); w[k] = rows; });
  return w;
}
const R0 = rng(7);
const pickRows = (n, k) => [...Array(n).keys()].sort(() => R0() - 0.5).slice(0, k).sort((a, b) => a - b);
const EIDX = pickRows(vin.length, 10), UIDX = pickRows(vout.length, 10);
// each block: [name, tensor, rows of flat offsets within the tensor, depth]; W1 and W2 drawn as 16 x 16 squares
const rowsOf = (idx) => idx.map((r) => [...Array(8).keys()].map((j) => r * 8 + j));
const grid = (R, Cn) => [...Array(R).keys()].map((i) => [...Array(Cn).keys()].map((j) => i * Cn + j));
const BLOCKS = [
  [["E", "E", rowsOf(EIDX), 0]],
  [["Wq", "Wq", grid(8, 8), -10], ["Wk", "Wk", grid(8, 8), 0], ["Wv", "Wv", grid(8, 8), 10]],
  [["Wo", "Wo", grid(8, 8), 0]],
  [["W1", "W1", grid(16, 16), 0]],
  [["W2", "W2", grid(16, 16), 0]],
  [["U", "U", rowsOf(UIDX), 0]],
];
const LABELS = [["E", "10,808 · embed"], ["Wq Wk Wv", "192 · attend"], ["Wo", "64 · mix"], ["W1", "288 · 32 units"], ["W2", "264 · back"], ["U", "12,204 · write"]];
const labText = LABELS.map((l) => l[1]);                    // the hero's label lines, which Run rewrites for E and U
function layoutNetwork(gap = 5) {
  const cells = [], centers = []; let x2 = 0, prev = 0;
  BLOCKS.forEach((group, gi) => {
    const half = Math.max(...group.map(([, , R]) => (R.length + R[0].length) / 2));
    x2 += prev + half + (prev ? gap : 0); prev = half;
    const u = x2 / 2;
    group.forEach(([name, ten, R, dep], bi) => {
      const cx = u + dep / 2, cz = -u + dep / 2, rows = R.length, cols = R[0].length;
      R.forEach((r, i) => r.forEach((k, j) => cells.push({ x: cx - cols / 2 + j + 0.5, z: cz - rows / 2 + i + 0.5, off: LAY.off[ten].at + k, name, block: name, i, j, group: gi })));
      centers.push({ name, group: gi, p: new THREE.Vector3(cx, 0, cz) });
    });
  });
  return { cells, centers };
}
const NET = layoutNetwork();
const BLOCKNAMES = [...new Set(NET.cells.map((c) => c.block))];
// Draw the network from a flat parameter array. Heights are scaled within each block; with `prev`, the bars that moved
// most since then glow sea glass.
function networkScene(v, H = 2.6, cells = NET.cells) {
  const m = barMesh(cells.length); v.root.add(m); v.net = m; v.cells = cells; v.vals = new Float32Array(cells.length);
  v.paint = (p, prev, hi) => {                          // hi(cell): a colour to light that bar with, { c, k } for a colour at strength k, or nothing
    const norm = {}, move = {};
    BLOCKNAMES.forEach((b) => { norm[b] = 1e-9; move[b] = 1e-9; });
    cells.forEach((c) => { norm[c.block] = Math.max(norm[c.block], Math.abs(p[c.off])); if (prev) move[c.block] = Math.max(move[c.block], Math.abs(p[c.off] - prev[c.off])); });
    cells.forEach((c, i) => {
      const val = p[c.off], n = norm[c.block]; v.vals[i] = val;
      wColor(val, n, tmpC);
      if (prev) tmpC.lerp(GLASS, 0.85 * Math.min(1, Math.abs(val - prev[c.off]) / move[c.block]));
      const h = hi && hi(c); if (h) { if (h.isColor) tmpC.lerp(h, 0.8); else tmpC.lerp(h.c, h.k); }
      setBar(m, i, c.x, c.z, Math.max(0.05, Math.min(1, Math.abs(val) / n) * H), 0.8, tmpC);
    });
    done(m);
  };
  v.paint(SHIPPED);
  return m;
}

// ---------------------------------------------------------------- hero: the network, running and training for real
const hero = addView("v-hero", { pad: 1.04 });
networkScene(hero, 2.6, NET.cells.map((c) => Object.assign({}, c)));   // its own cells: its E and U rows follow the pass
const passCenters = NET.centers.filter((c) => c.name !== "Wq" && c.name !== "Wv").map((c) => c.p.clone().setY(1.2));
const passPts = [];
for (let k = 0; k < passCenters.length - 1; k++) {
  const a = passCenters[k], b = passCenters[k + 1];
  const seg = arcPoints(a, b, 0.28 * a.distanceTo(b) * 0.5);
  passPts.push(...(k ? seg.slice(1) : seg));
}
const passTube = tube(passPts, 0.12, ORANGE); hero.root.add(passTube);
const dots = passCenters.map((p, k) => {
  const s = new THREE.Mesh(new THREE.SphereGeometry(k === passCenters.length - 1 ? 0.6 : 0.36, 20, 14), new THREE.MeshBasicMaterial({ color: k === passCenters.length - 1 ? ORANGE : WHITE }));
  s.position.copy(p); hero.root.add(s);
  if (k < passCenters.length - 1) { const ring = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.1, 8, 24), new THREE.MeshBasicMaterial({ color: ORANGE })); ring.position.copy(p); ring.lookAt(p.clone().add(DIR)); hero.root.add(ring); s.userData.ring = ring; }
  return s;
});
const pulse = new THREE.Mesh(new THREE.SphereGeometry(0.55, 20, 14), new THREE.MeshBasicMaterial({ color: ORANGE })); hero.root.add(pulse);
pulse.position.copy(passPts[0]);
const passLen = [0]; for (let i = 1; i < passPts.length; i++) passLen.push(passLen[i - 1] + passPts[i].distanceTo(passPts[i - 1]));
function passAt(u, out) {                                // the point a fraction u of the way along the pass
  const d = u * passLen[passLen.length - 1]; let i = 1; while (i < passLen.length - 1 && passLen[i] < d) i++;
  const f = (d - passLen[i - 1]) / Math.max(1e-9, passLen[i] - passLen[i - 1]);
  return out.copy(passPts[i - 1]).lerp(passPts[i], Math.min(1, Math.max(0, f)));
}
fit(hero);

// Run: the network runs the universal machine, which runs the current program (Collatz from 27 to begin with; picking
// a program anywhere on the page makes it current). Nothing moves until you press Step or ▶.
// The decoder is core.js's FastRun, which keeps the transcript bounded as the window moves along it. Each distinct
// window is computed by the network once, then looked up; every distinct window is checked against the universal
// machine's exact rule. The pass on show is computed again, live, so its rows and its confidence can be drawn.
const RVZ = window.TCT_RUNVIZ;
const RUN = { engine, who: "the network on this page", fr: null, cfgs: [], utmAtStep: 0, last: null,
  it: { id: "collatz", src: window.TCT_PROGRAMS.programs.find((p) => p.id === "collatz") } };
// the current program with its default input: its machine, a direct run of it (for the totals and the drawing's extent)
function heroProgram(it) {
  const sp = RVZ.SPECS[it.id], param = sp.param.value, m = RVZ.setup(it.id, it, param);
  Object.assign(RUN, { it, param, m, machine: TCT.parseMachine(m.start, m.rules), want: RVZ.direct(m.start, m.rules, m.tape) });
  RUN.fin = sp.finalOf ? sp.finalOf(RUN.want) : null;
  RUN.label = progTitle(it) + " · " + RVZ.show(it.id, param);
}
const progTitle = (it) => it.src.title === "Fibonacci word fractal" ? "Fibonacci word" : it.src.title;
const cfgOf = (c) => ({ state: c.simState, head: c.simHead, tape: c.simTape.join("").replace(/_+$/, "") });
function newRun() {
  const fr = RUN.fr = new TCT.FastRun(RUN.engine, RUN.machine, RUN.m.tape, (cfg, r) => { RUN.cfgs.push(cfgOf(cfg)); RUN.utmAtStep = r.utmSteps; });
  fr.onHalt = (cfg, r) => { RUN.cfgs.push(Object.assign(cfgOf(cfg), { state: r.verdict === "ACCEPT" ? "qa" : "qr", final: true })); };
  RUN.cfgs = [RUN.want[0]]; RUN.utmAtStep = 0; RUN.last = null;
}
heroProgram(RUN.it); newRun();
const isHead = (id) => RUN.engine.stateCode[id] > 0;
// Advance by one unit: a pass; passes up to and including the next whose window holds the head; a whole ring (one
// step of the universal machine); or a whole step of the program (hundreds to thousands of rings).
function advance(unit) {
  if (RUN.fr.verdict) newRun();
  const fr = RUN.fr, T = () => fr.T, u0 = fr.utmSteps;
  if (unit === "pass") fr.run(1);
  else if (unit === "head") { for (let n = 0; n < 1e6 && !fr.verdict; n++) { const k = fr.k, h = isHead(T()[k]) || isHead(T()[k + 1]) || isHead(T()[k + 2]); fr.run(1); if (h) break; } }
  else if (unit === "ring") { while (!fr.verdict && fr.utmSteps === u0) fr.run(1); }
  else { fr.stopEachStep = true; fr.run(1e9); fr.stopEachStep = false; }
  const at = fr.verdict ? fr.k : fr.k - 1, ids = [fr.T[at], fr.T[at + 1], fr.T[at + 2]];
  RUN.last = { at, ids, f: RUN.engine.net.forward(ids), unit };
  if (unit === "pass" || unit === "head") hero.anim = 0;
  render();
}

// the E rows of the three tokens read, among seven others; the U rows of the ten likeliest outputs, the chosen one lit
const LIGHT = C("#F05A1A"), READ = C("#F2955F");
function setRows(eRows, uRows) {
  hero.cells.forEach((c) => {
    if (c.block === "E") { c.row = eRows[c.i]; c.off = LAY.off.E.at + c.row * 8 + c.j; }
    if (c.block === "U") { c.row = uRows[c.i]; c.off = LAY.off.U.at + c.row * 8 + c.j; }
  });
}
function paintHero() {
  const p = RUN.who.startsWith("your") ? trainer.latest : SHIPPED, L = RUN.last;
  if (!L) { setRows(EIDX, UIDX); hero.paint(p); return; }
  const read = [...new Set(L.ids)], others = EIDX.filter((r) => !read.includes(r)).slice(0, 10 - read.length);
  const eRows = [...others.slice(0, 3), ...read, ...others.slice(3)].slice(0, 10);
  const lg = L.f.logits, top = [...lg.keys()].sort((a, b) => lg[b] - lg[a]).slice(0, 10);
  const uRows = [...top.slice(1, 5), top[0], ...top.slice(5)];
  setRows(eRows, uRows);
  const flow = passFlow(L.f);
  hero.paint(p, null, (c) => c.block === "E" ? (read.includes(c.row) ? READ : null) : c.block === "U" ? (c.row === L.f.best ? LIGHT : null) : flow(c));
}
// The middle blocks, lit by what each weight carried in this pass: the weight times the value it multiplied. Wq meets
// the newest token; Wk and Wv the three tokens, weighted by where that row's head looked; Wo attention's mix; W1 the
// residual stream, but only into units that fired (ReLU above 0); W2 the units' outputs, so units that stayed at 0 stay dark.
function passFlow(f) {
  const net = RUN.engine.net, dh = net.d / net.h;
  const mix = f.attention.map((a) => Array.from({ length: net.d }, (_, c) => a[0] * f.x[0][c] + a[1] * f.x[1][c] + a[2] * f.x[2][c]));
  const term = {
    Wq: (r, c) => net.Wq[r][c] * f.x[2][c],
    Wk: (r, c) => net.Wk[r][c] * mix[Math.floor(r / dh)][c],
    Wv: (r, c) => net.Wv[r][c] * mix[Math.floor(r / dh)][c],
    Wo: (r, c) => net.Wo[r][c] * f.y[c],
    W1: (r, c) => f.hidden[r] > 0 ? net.W1[r][c] * f.h1[c] : 0,
    W2: (r, c) => net.W2[r][c] * f.hidden[c],
  };
  const val = new Map(), max = {};
  hero.cells.forEach((c) => {
    const t = term[c.block]; if (!t) return;
    const o = LAY.off[c.block], k = c.off - o.at, v = Math.abs(t(Math.floor(k / o.cols), k % o.cols));
    val.set(c, v); max[c.block] = Math.max(max[c.block] || 1e-12, v);
  });
  return (c) => { const v = val.get(c); if (v === undefined) return null; const s = v / max[c.block]; return s < 0.3 ? null : { c: READ, k: 0.35 + 0.55 * (s - 0.3) / 0.7 }; };
}

// the transcript around the window: the ring being read, and the ring being written under it
const SIDE = 6;
function cell(t, cls, title) { return "<span class='t" + (cls ? " " + cls : "") + "'" + (title ? " title='" + esc(title) + "'" : "") + ">" + esc(t) + "</span>"; }
const tokCls = (t) => t === "#" ? "ring" : t.includes(":") ? "hd" : "";
function strip() {
  const fr = RUN.fr, T = fr.T, L = RUN.last, at = L ? L.at : fr.k, mid = at + 1;
  let top = "", bot = "", rs = mid, re = mid + 1;              // the reading ring, rs .. re: tokens outside it are dimmed
  while (rs > 0 && T[rs] !== fr.BID) rs--;
  while (re < fr.n && T[re] !== fr.BID) re++;
  for (let i = mid - SIDE; i <= mid + SIDE; i++) {
    if (i < 0 || i >= fr.n) { top += cell("", "more"); continue; }
    const t = vin[T[i]], w = i >= at && i <= at + 2;
    top += cell(t, (w ? "w " : "") + tokCls(t) + (i < rs || i >= re ? " o" : ""));
  }
  const len = L ? (fr.verdict ? 0 : RUN.engine.emitTyped[L.f.best].length) : 0, end = L ? fr.n : fr.n;
  const fresh = L && !fr.verdict ? len : 0, first = end - fresh;      // the tokens this pass wrote: first .. end
  for (let c = 0; c <= 2 * SIDE; c++) {
    const i = first - SIDE + c;
    if (c < SIDE) { bot += L && i >= 0 && i >= fr.ringStart ? cell(vin[T[i]], tokCls(vin[T[i]])) : cell("", "more"); continue; }
    if (c - SIDE < fresh) bot += cell(vin[T[first + c - SIDE]], "new");
    else bot += cell(c === SIDE && fr.verdict ? "halt" : "?", "q");
  }
  const ringLen = fr.n - fr.ringStart;
  $("rp-strip").innerHTML =
    "<div class='line'><span class='lab'>Reading · ring " + comma(fr.utmSteps) + "</span>" + top + "</div>" +
    "<div class='line'><span class='lab'>Writing · ring " + comma(fr.utmSteps + 1) + "</span>" + bot + "</div>";
  const el = $("rp-strip"), w = el.querySelector(".t.w");     // on a narrow screen, scroll the window into the middle
  if (w && el.scrollWidth > el.clientWidth) el.scrollLeft = w.offsetLeft + 1.5 * w.offsetWidth - el.clientWidth / 2;
  return ringLen;
}
// the universal machine's state: the newest head token in the transcript
function utmState() {
  const fr = RUN.fr; for (let i = fr.n - 1; i >= Math.max(0, fr.n - 3000); i--) if (isHead(fr.T[i])) return vin[fr.T[i]].split(":")[0];
  return "find";
}
function render() {
  const fr = RUN.fr, L = RUN.last, e = RUN.engine;
  strip();
  // 1. the network
  if (!L) {
    const ids = [fr.T[fr.k], fr.T[fr.k + 1], fr.T[fr.k + 2]];
    $("h-window").innerHTML = "<span class='tiny'>Next</span>" + ids.map((i) => "<span class='chip" + (isHead(i) ? " hd" : "") + "'>" + esc(vin[i]) + "</span>").join("") + "<span class='muted' style='font-size:14px; padding-left:8px'>Nothing computed yet. Press Step.</span>";
    $("rp-net").innerHTML = "Reads three tokens of the transcript, writes the next. The first window is outlined above; no pass has run.";
  } else {
    const toks = L.ids.map((i) => vin[i]), lg = L.f.logits;
    let mx = -Infinity; for (const x of lg) mx = Math.max(mx, x); let z = 0; for (const x of lg) z += Math.exp(x - mx);
    const sure = Math.round(Math.exp(lg[L.f.best] - mx) / z * 1000) / 10, wrote = D.emits[vout[L.f.best]].join(" "), exact = e.exact(L.ids), ok = wrote === exact;
    const head = L.ids.some(isHead), copy = wrote === toks[1], seen = e.trained.has(e.key(L.ids));
    $("h-window").innerHTML = "<span class='tiny'>Read</span>" + toks.map((t) => "<span class='chip" + (t.includes(":") ? " hd" : "") + "'>" + esc(t) + "</span>").join("") +
      "<span class='muted' style='padding:0 6px'>→</span><span class='tiny'>Wrote</span><span class='chip out'>" + esc(wrote || "(halt)") + "</span>";
    $("rp-net").innerHTML = "One forward pass: E looks up the rows of <code>" + toks.map(esc).join(" ") + "</code> (lit), attention and the MLP mix them (lit by what each weight carried this pass), and U scores all 1,356 outputs. " +
      "It chose <code>" + esc(wrote || "halt") + "</code>, " + sure + "% sure. " +
      (head ? (copy ? "The head is in the window, but this cell does not change." : "The head is in the window: this is where the universal machine's rule acts.") : "No head in the window, so the right answer is to copy the middle token.") +
      " The exact rule says <code>" + esc(exact || "halt") + "</code>: " + (ok ? "<span class='ok'>agrees</span>" : "<span class='bad'>disagrees</span>") + ". " +
      (seen ? "This window was in training." : "<b>This window was never in training.</b>");
  }
  // 2. the universal machine
  const st = utmState(), ringLen = fr.ringStart < 0 ? 0 : fr.n - fr.ringStart;
  $("rp-utm").innerHTML = "Each ring is one configuration of the universal machine; a full ring of passes is one of its steps. " +
    "It has taken <b>" + comma(fr.utmSteps) + "</b> step" + (fr.utmSteps === 1 ? "" : "s") + "; the ring being written has " + comma(ringLen) + " of its about " + comma(fr.encoded.length + 1) + " tokens. " +
    "State <code>" + esc(st) + "</code>: " + esc(TCT.phase(st)) + ".";
  // 3. the program: its configuration after its last step, read off the transcript, and its own drawing (runviz.js)
  const sp = RVZ.SPECS[RUN.it.id], g = RUN.cfgs[RUN.cfgs.length - 1], nSteps = RUN.cfgs.length - 1, total = RUN.want.length - 1;
  $("rp-prog-h").textContent = "3 · The program: " + RUN.label;
  $("rp-prog").innerHTML = (fr.verdict ? "<b>Halted</b> after all " + comma(nSteps) + " of its steps and " + comma(fr.passes) + " passes. Step again to start over. "
    : nSteps ? "It has taken <b>" + comma(nSteps) + "</b> of its " + comma(total) + " steps. " : "It has taken none of its " + comma(total) + " steps yet. ") +
    "State <code>" + esc(g.state) + "</code>, head on cell " + g.head + "." +
    (fr.verdict ? "" : " The universal machine is " + comma(fr.utmSteps - RUN.utmAtStep) + " step" + (fr.utmSteps - RUN.utmAtStep === 1 ? "" : "s") + " into the next.");
  const d = sp.draw(RUN.cfgs, RUN.param, !!fr.verdict, RUN.want.length, RUN.fin);
  $("rp-draw").innerHTML = d.svg; $("rp-big").textContent = nSteps ? d.big : "";
  $("rp-count").textContent = comma(fr.passes) + " passes · " + comma(fr.distinct) + " distinct windows, each checked against the exact rule: " + comma(fr.disagree) + " wrong · " + comma(fr.unseen) + " never in training";
  $("h-status").textContent = RUN.label + " · " + RUN.who + (playing ? "" : " · paused");
  paintHero(); labelRun(); runCard();
}

// Train: real full-batch Adam on the GPU (train-gpu.js), from the trainer's random start, on all 100,697 training steps,
// learning rate 0.003, until every training step is right: the recipe that trained the network on this page.
const trainer = { gpu: null, state: "idle", step: 0, t0: 0, n: 0, prev: null, caught: false, latest: null, curve: [] };
function trainNote(html) { const el = $("h-train-note"); el.innerHTML = html; el.hidden = false; }
// the training curve: steps right (0..100%) and loss (log scale, 10 down to 0.001) by Adam step, with the step at which
// the network on this page got every training step right (3,250) marked for comparison
const REF_STEP = 3250;
function drawCurve() {
  const svg = $("h-curve"); if (!svg) return;
  const pts = trainer.curve, last = pts.length ? pts[pts.length - 1].s : 0;
  const xmax = Math.max(3500, Math.ceil(last * 1.1 / 500) * 500);
  const X = (s) => (s / xmax * 1000).toFixed(1), YA = (f) => (200 - f * 200).toFixed(1);
  const YL = (l) => (200 - (Math.log10(Math.min(10, Math.max(l, 0.001))) + 3) / 4 * 200).toFixed(1);
  const every = Math.max(1, Math.ceil(pts.length / 1200)), use = pts.filter((p, i) => i % every === 0 || i === pts.length - 1);
  const line = (y, cls) => "<line x1='0' x2='1000' y1='" + y + "' y2='" + y + "' class='" + cls + "' vector-effect='non-scaling-stroke'/>";
  svg.innerHTML = line(0, "g") + line(100, "g") + line(200, "g0") +
    "<line x1='" + X(REF_STEP) + "' x2='" + X(REF_STEP) + "' y1='0' y2='200' class='ref' vector-effect='non-scaling-stroke'/>" +
    (use.length ? "<polyline class='loss' vector-effect='non-scaling-stroke' points='" + use.map((p) => X(p.s) + "," + YL(p.l)).join(" ") + "'/>" +
      "<polyline class='acc' vector-effect='non-scaling-stroke' points='" + use.map((p) => X(p.s) + "," + YA(p.f)).join(" ") + "'/>" : "");
  $("h-curve-ref").style.left = (REF_STEP / xmax * 100) + "%"; $("h-curve-ref").classList.toggle("left", REF_STEP / xmax > 0.6);
  $("h-curve-max").textContent = comma(xmax);
}
// the Learning frame further down: a button to start training until a run is under way, then what it is waiting for
function learningFrame() {
  if (!$("f1-train")) return;
  const idle = trainer.state === "idle";
  $("f1-train").hidden = !idle;
  $("f1-wait").textContent = idle ? "Waiting for your run" : "Training · caught at half the steps right";
}
async function startTraining() {
  if (trainer.state === "running") return;
  if (trainer.state === "paused") { trainer.state = "running"; learningFrame(); trainLoop(); return; }
  if (!navigator.gpu) { trainNote("Training runs on WebGPU, which this browser does not offer. Chrome or Edge on a recent computer will. The network on this page was trained the same way."); return; }
  trainer.state = "starting"; learningFrame();
  trainNote("Starting the GPU, and loading the 100,697 training steps…");
  try {
    if (!trainer.gpu) {
      trainer.gpu = await window.TrainGPU.create(LAY, { fusedHead: true });
      const ids = D.trainIds, n = ids.length / 3, labels = new Uint32Array(n), index = new Map(vout.map((v, i) => [v, i]));
      for (let k = 0; k < n; k++) { const ex = engine.exact([ids[3 * k], ids[3 * k + 1], ids[3 * k + 2]]); labels[k] = index.get(ex === "" ? "<halt>" : ex); }
      trainer.gpu.setData({ steps: ids, labels }, Array.from({ length: n }, (_, i) => i));
      trainer.n = n;
    }
    trainer.gpu.setParams(INIT); trainer.step = 0; trainer.prev = INIT; trainer.caught = false; trainer.t0 = performance.now(); trainer.curve = []; drawCurve();
    hero.paint(INIT);
    trainer.state = "running"; trainNote("Training on your GPU: full-batch Adam on all 100,697 training steps…"); trainLoop();
  } catch (err) { trainer.state = "idle"; learningFrame(); trainNote("The GPU could not start training here (" + esc(err.message || err) + "). The network on this page was trained the same way."); }
}
async function trainLoop() {
  const g = trainer.gpu;
  while (trainer.state === "running") {
    const r = await g.step(0.003); trainer.step++;
    const frac = r.correct / trainer.n;
    trainer.curve.push({ s: trainer.step, f: frac, l: r.loss });
    if (trainer.step % 3 === 0 || r.correct === trainer.n) {
      if (heroMode === "train") drawCurve();
      const p = await g.params(); trainer.latest = p;
      if (heroMode === "train") hero.paint(p, trainer.prev);
      trainer.prev = p;
      if (!trainer.caught && frac >= 0.5) { trainer.caught = true; frames[1].paint(p); frames[1].shown = true; $("f1-over").hidden = true; $("f1-cap").textContent = "Caught from your run at Adam step " + comma(trainer.step) + ", with half the training steps right."; }
    }
    const secs = (performance.now() - trainer.t0) / 1000;
    trainer.last = r;
    if (heroMode === "train") {
      $("h-fill").style.width = (frac * 100).toFixed(1) + "%";
      $("h-status").textContent = "Adam step " + comma(trainer.step) + " · loss " + r.loss.toFixed(4) + " · " + secs.toFixed(0) + " s";
      $("h-window").innerHTML = "<span class='tiny'>Training steps right</span><span class='num' style='font-size:24px'>" + comma(r.correct) + "</span><span class='muted'>of " + comma(trainer.n) + " · " + (frac * 100).toFixed(1) + "%</span>";
      if (trainer.step % 10 === 0 && r.correct < trainer.n) trainNote("Training on your GPU: full-batch Adam on all 100,697 training steps, about " + (trainer.step / secs).toFixed(1) +
        " steps a second. The network on this page needed 3,250 steps to get every one right. It keeps training while you read on.");
    }
    if (r.correct === trainer.n) {
      trainer.state = "done";
      const p = trainer.latest;
      RUN.engine = new TCT.Engine(Object.assign({}, D, { weights: toWeights(p) })); RUN.who = "your network, trained just now"; newRun();
      $("h-play").innerHTML = PLAY;
      trainNote("Every training step right at Adam step " + comma(trainer.step) + ", after " + secs.toFixed(0) + " seconds on your GPU. Switch to <b>Run</b>: the pass now runs the network you just trained, checked against the exact rule.");
    } else if (trainer.step >= 8000) { trainer.state = "done"; trainNote("Stopped at Adam step 8,000 with " + comma(r.correct) + " of " + comma(trainer.n) + " training steps right."); }
  }
}

let heroMode = "run", playing = false, heroT = 0;
const PERIOD = { pass: 1.1, head: 1.1, ring: 0.6, step: 0.5 }, PULSE = 0.9;
const PLAY = '<svg width="12" height="14" viewBox="0 0 12 14"><path d="M1 1 L11 7 L1 13 Z" fill="#fff"/></svg>',
      PAUSE = '<svg width="12" height="14" viewBox="0 0 12 14"><path d="M1 1 H4.5 V13 H1 Z M7.5 1 H11 V13 H7.5 Z" fill="#fff"/></svg>';
const unit = () => $("h-unit").value;
function setPlaying(on) { playing = on; $("h-play").innerHTML = on ? PAUSE : PLAY; $("h-play").setAttribute("aria-label", on ? "Pause" : "Play"); if (heroMode === "run") $("h-status").textContent = RUN.label + " · " + RUN.who + (on ? "" : " · paused"); }
hero.anim = 1;                                           // where the pulse is along the pass, 0 .. 1; it rests at U
hero.tick = (t, dt) => {
  if (heroMode !== "run") return;
  if (playing) { heroT += dt; if (heroT >= PERIOD[unit()]) { heroT = 0; advance(unit()); } }
  if (hero.anim < 1) hero.anim = Math.min(1, hero.anim + dt / PULSE);
  passAt(RUN.last ? hero.anim : 0, pulse.position);
};
function setMode(m) {
  heroMode = m;
  $("mode-run").setAttribute("aria-pressed", m === "run"); $("mode-train").setAttribute("aria-pressed", m === "train");
  [passTube, pulse, ...dots, ...dots.map((d) => d.userData.ring).filter(Boolean)].forEach((o) => { o.visible = m === "run"; });
  $("h-run").hidden = m !== "run"; $("h-step").hidden = m !== "run"; $("h-unit").parentNode.hidden = m !== "run"; $("h-prog").parentNode.hidden = m !== "run"; $("h-trainbar").hidden = m === "run";
  if ($("h-trainpanel")) { $("h-trainpanel").hidden = m === "run"; if (m !== "run") drawCurve(); }
  if (m === "run") {
    setPlaying(playing);
    $("h-train-note").hidden = trainer.state !== "done";
    render();
  } else {
    setPlaying(false);
    $("h-play").innerHTML = trainer.state === "running" ? PAUSE : PLAY; $("h-play").setAttribute("aria-label", trainer.state === "running" ? "Pause training" : "Train");
    setRows(EIDX, UIDX); labelTrain();
    if (trainer.latest) hero.paint(trainer.latest); else hero.paint(INIT);
    if (trainer.state === "idle") {
      $("h-status").textContent = "A fresh network, from a random start · press ▶ to train";
      $("h-window").innerHTML = "<span class='muted' style='font-size:14px'>Full-batch Adam on your GPU, on all 100,697 training steps, until every one is right.</span>";
      $("h-fill").style.width = "0%";
      trainNote("These are the trainer's random starting weights. Press ▶ and gradient descent trains them, here, on your GPU; bars glow green where the weights are moving.");
    }
  }
}
$("mode-run").onclick = () => setMode("run"); $("mode-train").onclick = () => setMode("train");
if ($("f1-train")) $("f1-train").onclick = () => {      // up to the stage, in Train mode, already training
  $("v-hero").closest(".stage").scrollIntoView({ behavior: "smooth", block: "start" });
  setMode("train"); startTraining();
  $("h-play").innerHTML = PAUSE; $("h-play").setAttribute("aria-label", "Pause training");
};
$("h-play").onclick = () => {
  if (heroMode === "train") {
    if (trainer.state === "running") { trainer.state = "paused"; $("h-play").innerHTML = PLAY; trainNote("Paused at Adam step " + comma(trainer.step) + ". Press ▶ to go on."); }
    else { startTraining(); $("h-play").innerHTML = PAUSE; }
    return;
  }
  heroT = PERIOD[unit()]; setPlaying(!playing);
};
$("h-step").onclick = () => { if (heroMode !== "run") return; setPlaying(false); advance(unit()); };
$("h-unit").onchange = () => { heroT = 0; };
hoverable(hero, () => [hero.net], (m, i) => {
  const c = hero.cells[i], v = hero.vals[i].toFixed(3);
  if (c.block === "E") return "E row of " + vin[c.row] + ", [" + c.j + "] = " + v + (RUN.last && heroMode === "run" && RUN.last.ids.includes(c.row) ? "\nread by this pass" : "");
  if (c.block === "U") { const o = D.emits[vout[c.row]].join(" ") || "halt"; return "U row of " + o + ", [" + c.j + "] = " + v + (RUN.last && heroMode === "run" ? "\nscore " + RUN.last.f.logits[c.row].toFixed(2) + (c.row === RUN.last.f.best ? ", the one chosen" : "") : ""); }
  return c.name + "[" + c.i + "][" + c.j + "] = " + v;
});
// the block labels, placed under the blocks wherever the camera puts them
function placeLabels() {
  const el = $("h-labels"); if (!hero.aspect) return requestAnimationFrame(placeLabels);
  el.innerHTML = "";
  const groups = [0, 1, 2, 3, 4, 5].map((g) => NET.centers.filter((c) => c.group === g));
  groups.forEach((cs, g) => {
    const p = cs[0].p.clone(); if (cs.length > 1) p.copy(cs[1].p);
    p.project(hero.camera);
    const s = document.createElement("span"); s.style.left = ((p.x + 1) / 2 * 100) + "%";
    s.innerHTML = "<b" + (g === 5 ? " style='color:var(--accent)'" : "") + ">" + LABELS[g][0] + "</b><small>" + esc(labText[g]) + "</small>";
    el.appendChild(s);
  });
}
function labelRun() {
  const L = RUN.last; labText[0] = L ? "3 rows read, lit" : LABELS[0][1]; labText[5] = L ? "chose " + (D.emits[vout[L.f.best]].join(" ") || "halt") : LABELS[5][1];
  labText[3] = L ? L.f.hidden.filter((h) => h > 0).length + " of " + L.f.hidden.length + " units fired" : LABELS[3][1]; placeLabels();
}
function labelTrain() { labText[0] = LABELS[0][1]; labText[3] = LABELS[3][1]; labText[5] = LABELS[5][1]; placeLabels(); }
placeLabels(); addEventListener("resize", () => { hero.aspect = 0; requestAnimationFrame(() => requestAnimationFrame(placeLabels)); });
setMode("run");                                         // Run, paused, on the first window: nothing moves until Step or ▶

// the Run column: the two rings around the hero's pass, live, whatever program it is running; and what running every
// program once cost the network (program-stats.js, generated by build_stats.js)
function runCard() {
  const fr = RUN.fr, T = fr.T, L = RUN.last, at = L ? L.at : fr.k;
  const tok = (t, cls) => "<span class='tok " + (cls || "") + "'>" + esc(t) + "</span>", cl = (i) => isHead(T[i]) ? "hd" : "";
  let read = "", write = "";
  for (let i = Math.max(0, at - 4); i < at; i++) read += tok(vin[T[i]], cl(i));
  read += "<span class='winbox'>" + [0, 1, 2].map((d) => tok(vin[T[at + d]], cl(at + d) || "w")).join("") + "</span>";
  for (let i = at + 3; i < Math.min(fr.n, at + 5); i++) read += tok(vin[T[i]], cl(i));
  const len = L && !fr.verdict ? RUN.engine.emitTyped[L.f.best].length : 0, first = fr.n - len;
  if (L) for (let i = Math.max(fr.ringStart, first - 4); i < first; i++) write += tok(vin[T[i]], cl(i));
  for (let i = first; i < fr.n; i++) write += tok(vin[T[i]], "out");
  write += tok("?", "q");
  $("run-rings").innerHTML = "<p class='tiny run-cap'>Live from the scene above · " + esc(RUN.label) + " · " + (L ? "pass " + comma(fr.passes) : "before the first pass") + "</p>" +
    "<div class='ring-row'><span class='tiny'>Ring " + comma(fr.utmSteps) + "</span>" + read + "</div><div class='ring-row'><span class='tiny'>Ring " + comma(fr.utmSteps + 1) + "</span>" + write + "</div>";
}
{
  const A = window.TCT_STATS.all;
  $("rc-passes").textContent = (A.passes / 1e9).toFixed(1) + " billion";
  $("rc-passes-l").textContent = "forward passes to run all " + A.programs + " programs once";
  $("rc-unseen").textContent = comma(A.unseen);
  $("rc-wrong").textContent = comma(A.wrong);
}

// the training frames: the trainer's real random start, a moment caught from your own run, and the trained network
const frames = ["v-f0", "v-f1", "v-f2"].map((id, k) => {
  const v = addView(id, { pad: 1.02 }); networkScene(v); v.paint(k === 0 ? INIT : SHIPPED); fit(v);
  if (k === 1) v.net.visible = false;
  return v;
});
frames[1].paint = ((orig) => (p) => { frames[1].net.visible = true; orig(p); })(frames[1].paint);

// ---------------------------------------------------------------- the training data, from a real random machine
const SAMPLE = ["s0 0 -> s2 0 R", "s0 1 -> qa 0 R", "s0 _ -> s1 0 R", "s1 0 -> s0 1 S", "s1 1 -> s1 _ S", "s2 1 -> s2 0 R", "s2 _ -> s3 1 L",
  "s3 0 -> s0 1 L", "s3 1 -> s3 1 L", "s3 _ -> s3 0 R", "s4 0 -> s1 _ L", "s4 1 -> s4 1 R", "s4 _ -> s1 0 L"];
function entriesOf(enc) {                                 // the description, split into one entry per state, and the tape
  const d = enc.indexOf("$"), prog = enc.slice(0, d), tape = enc.slice(d + 1), out = [];
  prog.split(/(?=[*|])/).forEach((e) => { const sep = e[0], tag = e[1], rules = e.slice(2).split(";").filter((x) => x).map((x) => x + ";"); out.push({ sep, tag, rules }); });
  return { entries: out, tape };
}
{
  $("td-rules").textContent = SAMPLE.join("\n");
  const m = TCT.parseMachine("s0", SAMPLE), enc = TCT.utms[2].encode(m, "_10_"), E = entriesOf(enc);
  $("td-enc").innerHTML = E.entries.map((e) => esc(e.sep + e.tag + " " + e.rules.join(" "))).join("\n") + "\n<span style='color:var(--accent)'>$" + esc(E.tape) + "</span>";
  $("td-enc-note").textContent = "One line per state: what to write, which way to move, and the next state's number in binary. Then $ and the machine's own tape. " + enc.length + " cells.";
  const run = new TCT.Run(engine, m, "_10_"), first = new Map();
  while (!run.verdict) { const ids = run.window(), k = engine.key(ids); if (!first.has(k)) first.set(k, { w: ids.map((i) => vin[i]), out: engine.exact(ids) }); run.step(false); }
  $("td-run-note").textContent = "Each row is the tape after one step; the orange cell is the head, here scanning right. This machine stops after " + run.simSteps + " steps of its own, which take the universal machine " + comma(run.utmSteps) + ".";
  const all = [...first.values()], one = (f) => all.find(f);
  const rows = [one((x) => x.w[0].startsWith("find:")), one((x) => x.w[1].startsWith("find:")), one((x) => !x.w.some((t) => t.includes(":"))),
                one((x) => x.w[1].startsWith("go_")), one((x) => x.w[1].startsWith("jmp"))].filter(Boolean);
  $("td-wins").innerHTML = "<div><span>L</span><span>C</span><span>R</span><span></span><span>WRITES</span></div>" + rows.map((r) =>
    "<div>" + r.w.map((t) => "<span" + (t.includes(":") ? " class='h'" : "") + ">" + esc(t) + "</span>").join("") + "<span class='arrow'>→</span><span" + (r.out.includes(":") ? " class='h'" : "") + ">" + esc(r.out) + "</span></div>").join("");
  $("td-wins-note").textContent = "Three tokens of the transcript, and the token that goes in the middle one's place next. This run gives " + comma(first.size) + " different ones.";
  let svg = "";
  for (let r = 0; r < 14; r++) { svg += "<rect x='0' y='" + r * 14 + "' width='202' height='8' rx='2' fill='#E3E8EB'/><rect x='" + (6 + r * 14) + "' y='" + r * 14 + "' width='8' height='8' rx='2' fill='#F05A1A'/>"; }
  $("td-rings").innerHTML = svg;
  // the coverage curve, from the collection log
  const pts = [[1, 2665], [10, 29851], [30, 40711], [100, 81187], [250, 89761], [500, 94178], [1000, 96577], [2000, 98215], [3000, 98935], [4000, 99559]];
  const X = (n) => 48 + Math.log10(n) / Math.log10(4000) * 600, Y = (v) => 196 - v / 105000 * 176;
  let ch = "<g stroke='#E3E8EB'>" + [0, 50000, 100000].map((v) => "<line x1='48' x2='740' y1='" + Y(v) + "' y2='" + Y(v) + "'/>").join("") + "</g>";
  ch += "<g font-family='IBM Plex Mono' font-size='11' fill='#5A6A78'>" + [[0, "0"], [50000, "50k"], [100000, "100k"]].map(([v, s]) => "<text x='0' y='" + (Y(v) + 4) + "'>" + s + "</text>").join("") +
        [1, 10, 100, 1000, 4000].map((n) => "<text x='" + (X(n) - 4) + "' y='216'>" + comma(n) + "</text>").join("") + "</g>";
  ch += "<polyline fill='none' stroke='#3E9C8C' stroke-width='2.5' stroke-linejoin='round' points='" + pts.map(([n, v]) => X(n).toFixed(1) + "," + Y(v).toFixed(1)).join(" ") + "'/>";
  ch += "<line x1='648' y1='" + Y(99559) + "' x2='704' y2='" + Y(100697) + "' stroke='#0B1F33' stroke-width='2.5' stroke-dasharray='3 3'/><circle cx='704' cy='" + Y(100697) + "' r='5' fill='#0B1F33'/>";
  ch += "<text x='566' y='14' font-family='IBM Plex Mono' font-size='11' fill='#0B1F33'>+200 TOURS → 100,697</text>";
  $("td-chart").innerHTML = ch;
}

// ---------------------------------------------------------------- program anatomy
const PR = window.TCT_PROGRAMS.programs, CU = window.TCT_CURVES.curves, prog = (id) => PR.find((p) => p.id === id), curve = (id) => CU.find((c) => c.id === id);
const BB4 = ["A _ -> B 1 R", "A 1 -> B 1 L", "B _ -> A 1 L", "B 1 -> C _ L", "C _ -> qa 1 R", "C 1 -> D 1 L", "D _ -> D 1 R", "D 1 -> A _ R"];
const SIEVE = window.TCT_PRIMES.sieve(1680);
const LIST = [
  ["Numbers", [
    { id: "counter", src: prog("counter"), input: prog("counter").input(3), shown: "3 bits, all 0" },
    { id: "sort", src: prog("sort"), input: prog("sort").input("1011"), shown: "1011" },
    { id: "count", src: prog("count"), input: prog("count").input(3), shown: "three 1s" },
    { id: "add", src: prog("add"), input: prog("add").input("3+2"), shown: "3 + 2" },
    { id: "collatz", src: prog("collatz"), input: prog("collatz").input(6), shown: "n = 6" },
    { id: "fib", src: prog("fib"), input: prog("fib").input(4), shown: "4 bits" }]],
  ["Drawings", [
    { id: "dragon", src: curve("dragon"), input: "_000", shown: "3 bits" },
    { id: "koch", src: curve("koch"), input: "_000", shown: "3 bits" },
    { id: "fibword", src: curve("fibword"), input: "_000", shown: "3 bits" }]],
  ["Records and scale", [
    { id: "bb4", src: { title: "Busy beaver", start: "A", rules: BB4, blurb: "The four-state busy beaver: the four-state machine that runs longest before stopping, 107 steps, leaving 13 ones." }, input: "_", shown: "a blank tape" },
    { id: "sieve", src: { title: "Prime sieve", start: "s", rules: SIEVE.rules, blurb: "The sieve of Eratosthenes up to 1,680, explained in full in the next section." }, input: SIEVE.tape, shown: "1,681 cells", big: true }]],
];
const picker = $("picker"); let current = null;
LIST.forEach(([grp, items]) => {
  const g = document.createElement("div"); g.className = "grp"; g.textContent = grp; picker.appendChild(g);
  items.forEach((it) => {
    it.m = TCT.parseMachine(it.src.start, it.src.rules);
    const b = document.createElement("button"); b.type = "button";
    b.innerHTML = "<span>" + esc(it.src.title === "Fibonacci word fractal" ? "Fibonacci word" : it.src.title) + "</span><span class='n'><span>" + it.m.order.length + "</span><span>" + it.src.rules.length + "</span></span>";
    b.onclick = () => choose(it); it.btn = b; picker.appendChild(b);
  });
});
const LETW = { a: "0", b: "1", c: "_" }, MOVE = { L: "move left", R: "move right", S: "stay" };
function choose(it) {
  current = it;
  LIST.forEach(([, items]) => items.forEach((x) => x.btn.setAttribute("aria-pressed", x === it)));
  const m = it.m, enc = TCT.utms[2].encode(m, it.input), E = entriesOf(enc), big = !!it.big;
  $("p-title").textContent = it.src.title; $("p-blurb").textContent = it.src.blurb || "";
  const syms = ["0", "1", "_"], rows = [], states = big ? m.order.slice(0, 12) : m.order;
  states.forEach((q) => syms.forEach((s) => { const r = m.rules[q + "\u0000" + s]; if (r) rows.push({ q, s, r }); }));
  $("p-rules").innerHTML = "<div class='r th'><span>STATE</span><span>READ</span><span></span><span>NEXT</span><span>WRITE</span><span>MOVE</span></div>" +
    rows.map((x) => "<div class='r' data-k='" + esc(x.q + "|" + x.s) + "'><span><b>" + esc(x.q) + "</b></span><span>" + x.s + "</span><span>→</span><span>" + esc(x.r.next) + "</span><span>" + x.r.write + "</span><span>" + x.r.move + "</span></div>").join("") +
    (big ? "<div class='r'><span class='muted'>… " + comma(it.src.rules.length - rows.length) + " more rules</span></div>" : "");
  $("p-how").textContent = it.src.how || "Each line is a rule, in the order state, read, next state, write, move. qa means stop.";
  const shownEntries = big ? E.entries.slice(0, 12) : E.entries;
  $("p-grid").innerHTML = "<div class='r th'><span>STATE · NUMBER</span><span>READS 0</span><span>READS 1</span><span>READS _</span></div>" +
    shownEntries.map((e, i) => "<div class='r'><span><b>" + esc(m.order[i]) + "</b><small>#" + i + " · " + esc(e.sep + e.tag) + "</small></span>" +
      e.rules.map((r, j) => "<span><span class='cell" + (r === "H;" ? " nil" : "") + "' data-k='" + esc(m.order[i] + "|" + syms[j]) + "'>" + esc(r) + "</span></span>").join("") + "</div>").join("") +
    (big ? "<div class='r'><span class='muted'>… " + comma(E.entries.length - 12) + " more states</span></div>" : "");
  // the whole tape: the program, entry by entry, then its input
  const cells = (str, cls) => [...str].map((ch, k) => "<i class='" + (cls ? cls(ch, k) : "") + "'>" + esc(ch) + "</i>").join("");
  const showE = big ? E.entries.slice(0, 3) : E.entries;
  let tape = showE.map((e, i) => "<div class='entry' data-e='" + i + "'><div class='cells'>" + cells(e.sep + e.tag + e.rules.join(""), (ch, k) => k === 0 && ch === "*" ? "mark" : "") +
    "</div><div class='brk'></div><small>" + esc(m.order[i]) + (i === 0 ? " · current state" : "") + "</small></div>").join("");
  if (big) tape += "<div class='entry'><div class='cells'><i style='width:auto; padding:5px 8px'>… " + (E.entries.length - 3) + " more states …</i></div><div class='brk' style='border-color:transparent'></div><small></small></div>";
  const tapeShown = big ? E.tape.slice(0, 24) : E.tape;
  tape += "<div class='entry'><div class='cells'><i class='mark'>$</i>" + cells(tapeShown, (ch) => "OIU".includes(ch) ? "hd" : "") + (big ? "<i style='width:auto; padding:5px 8px'>…</i>" : "") + "</div><div class='brk'></div><small>input</small></div>";
  $("p-tape").innerHTML = tape;
  $("p-tape-label").textContent = "What the universal machine reads · " + comma(enc.length) + " cells";
  $("p-tape-input").textContent = "The program, then its input: " + it.shown;
  rbChoose(it); setHeroProgram(it);
  // trace one rule: in the last states, the first that jumps to a state numbered 2 or more
  let pickK = null;
  for (const q of [...states].reverse()) { for (const s of syms) { const r = m.rules[q + "\u0000" + s]; if (r && m.order.indexOf(r.next) >= 2) { pickK = q + "|" + s; break; } } if (pickK) break; }
  trace(pickK || (rows[0] && rows[0].q + "|" + rows[0].s));
}
function trace(k) {
  if (!k || !current) return;
  const m = current.m, [q, s] = k.split("|"), r = m.rules[q + "\u0000" + s], i = m.order.indexOf(q), syms = ["0", "1", "_"];
  document.querySelectorAll("#p-rules .r").forEach((e) => e.classList.toggle("on", e.dataset.k === k));
  document.querySelectorAll("#p-grid .cell").forEach((e) => e.classList.toggle("on", e.dataset.k === k));
  const enc = TCT.utms[2].encode(m, current.input), E = entriesOf(enc), code = E.entries[i] ? E.entries[i].rules[syms.indexOf(s)] : "";
  // the tape: light the four cells of this rule
  document.querySelectorAll("#p-tape .entry").forEach((el) => {
    const ei = +el.dataset.e; if (el.dataset.e === undefined) return;
    const e = E.entries[ei]; let off = 2;
    for (let j = 0; j < syms.indexOf(s) && ei === i; j++) off += e.rules[j].length;
    el.querySelectorAll(".cells i").forEach((c, ci) => { c.classList.toggle("on", ei === i && ci >= off && ci < off + code.length); });
  });
  if (!r || !code) { $("p-decode").innerHTML = ""; return; }
  const target = code.slice(2, -1), w = code[0], mv = code[1];
  const tgt = target === "A" ? "accept: stop" : target === "H" ? "halt" :
    "go to state #" + m.order.indexOf(r.next) + " (" + r.next + "): " + (m.order.indexOf(r.next)).toString(2) + " in binary, one letter per 1 bit" + (target ? "" : "; no letters means #0");
  $("p-decode").innerHTML = "<div><span class='tiny' style='color:var(--accent)'>The traced rule</span><span class='mono' style='font-size:13px'>" + esc(q + " " + s + " -> " + r.next + " " + r.write + " " + r.move) + "</span></div>" +
    "<div><b>" + w + "</b>write " + LETW[w] + " (a, b, c are 0, 1, blank)</div><div><b>" + mv + "</b>" + MOVE[mv] + "</div>" +
    "<div style='flex:1.4'><b>" + esc(target || "·") + "</b>" + esc(tgt) + "</div>";
}
document.addEventListener("pointerover", (e) => { const t = e.target.closest("#p-rules .r[data-k], #p-grid .cell[data-k]"); if (t) trace(t.dataset.k); });
// ---------------------------------------------------------------- run it on the network
// The picked program, run for real: the network (the one on this page, or yours once you have trained one) decodes the
// universal machine through the delta decoder (core.js DeltaRun), and after every step of the program its configuration
// is read off the transcript, checked against a direct run of the rules, and drawn (runviz.js).
const RV = window.TCT_RUNVIZ;
const RB = { it: null, param: null, run: null, got: [], want: [], fin: null, matched: 0, playing: false, fast: false, owed: 0, last: 0, t: 0, ms: 0 };
const PLAYI = '<svg width="10" height="12" viewBox="0 0 10 12"><path d="M1 1 L9 6 L1 11 Z" fill="#fff"/></svg>',
      PAUSEI = '<svg width="10" height="12" viewBox="0 0 10 12"><path d="M1 1 H3.8 V11 H1 Z M6.2 1 H9 V11 H6.2 Z" fill="#fff"/></svg>';
function rbField(it) {
  const p = RV.SPECS[it.id].param, el = $("rb-pfield");
  $("rb-plabel").textContent = p.label;
  el.innerHTML = p.options ? "<select id='rb-param'>" + p.options.map(([v, l]) => "<option value='" + v + "'>" + esc(l) + "</option>").join("") + "</select>"
    : "<input id='rb-param' " + (p.pattern ? "type='text' size='" + Math.min(18, String(p.value).length + 2) + "'" : "type='number' min='" + p.min + "' max='" + p.max + "'") + " spellcheck='false'>";
  $("rb-param").value = p.value;
  $("rb-param").onchange = () => rbReset();
}
function rbReset() {
  rbStop();
  const it = RB.it, sp = RV.SPECS[it.id];
  try { RB.param = RV.parse(it.id, $("rb-param").value.trim()); $("rb-hint").hidden = true; }
  catch (err) { $("rb-hint").textContent = "Input: " + err.message; $("rb-hint").hidden = false; return; }
  const m = RV.setup(it.id, it, RB.param);
  RB.want = RV.direct(m.start, m.rules, m.tape);
  RB.fin = sp.finalOf ? sp.finalOf(RB.want) : null;           // the whole run's extent, so the drawing does not rescale as it grows
  RB.got = [RB.want[0]]; RB.matched = 1; RB.ms = 0;
  RB.run = new TCT.DeltaRun(RUN.engine, TCT.parseMachine(m.start, m.rules), m.tape, (r) => {
    const g = { state: r.verdict ? (r.verdict === "ACCEPT" ? "qa" : "qr") : r.simState(), head: r.simHead(), tape: r.simTape().join("").replace(/_+$/, "") };
    const w = RB.want[RB.got.length];
    if (w && g.state === w.state && g.tape === w.tape && (w.final || g.head === w.head)) RB.matched++;
    RB.got.push(g);
  });
  rbRender();
}
function rbStep() {                                       // one step of the program; false once it has halted
  const r = RB.run; if (!r || r.verdict) return false;
  const n = RB.got.length, t0 = performance.now();
  r.stopEachStep = true;
  while (!r.verdict && RB.got.length === n) r.run(1e7);
  r.stopEachStep = false;
  RB.ms += performance.now() - t0;
  return true;
}
function rbTick(now) {
  if (!RB.playing) return;
  const t0 = performance.now(), dt = Math.min(0.25, (now - RB.last) / 1000); RB.last = now;
  if (RB.fast) { while (!RB.run.verdict && performance.now() - t0 < 28) rbStep(); }
  else {                                                   // watch: the whole run in about 30 seconds, at least 4 steps a second
    RB.owed += dt * Math.max(4, RB.want.length / 30);
    while (RB.owed >= 1 && !RB.run.verdict && performance.now() - t0 < 28) { rbStep(); RB.owed--; }
  }
  rbRender();
  if (RB.run.verdict) { rbStop(); rbRender(); return; }
  RB.t = requestAnimationFrame(rbTick);
}
function rbPlay() {
  if (!RB.run) return;
  if (RB.run.verdict) rbReset();
  RB.playing = true; RB.owed = 1; RB.last = performance.now(); rbButtons(); RB.t = requestAnimationFrame(rbTick);
}
function rbStop() { RB.playing = false; cancelAnimationFrame(RB.t); rbButtons(); }
function rbButtons() {
  const done = RB.run && RB.run.verdict;
  $("rb-play").innerHTML = RB.playing ? PAUSEI + "Pause" : PLAYI + (done ? "Run again" : RB.got.length > 1 ? "Go on" : "Run");
  $("rb-watch").setAttribute("aria-pressed", !RB.fast); $("rb-fast").setAttribute("aria-pressed", RB.fast);
}
function rbRender() {
  const it = RB.it, sp = RV.SPECS[it.id], r = RB.run, done = !!(r && r.verdict), steps = RB.got.length - 1, total = RB.want.length - 1;
  const d = sp.draw(RB.got, RB.param, done, RB.want.length, RB.fin);
  $("rb-svg").innerHTML = d.svg;
  $("rb-big").textContent = d.big; $("rb-sub").textContent = steps ? d.sub : "Nothing has run yet. Press Run, or Step to take one step at a time.";
  $("rb-what").textContent = (done ? "Finished · " : RB.playing ? "Running · " : steps ? "Paused · " : "Ready · ") + "read off the transcript, step by step";
  $("rb-steps").textContent = comma(steps);
  $("rb-steps-l").textContent = "of " + comma(total) + " steps of the program" + (it.id === "bb4" ? " (the first " + (RV.CHAMPS[RB.param].pad + 1) + " walk right, to make room on the one-way tape)" : "") + (r && RB.ms > 300 && steps ? " · " + comma(Math.round(steps / RB.ms * 1000)) + " a second" : "");
  $("rb-passes").textContent = comma(r ? r.passes : 0);
  $("rb-passes-l").textContent = "forward passes" + (steps ? " · " + comma(Math.round(r.passes / steps)) + " per step" : "") + (r && r.read ? " · the delta decoder computed " + comma(r.read) + " of them, the rest are copies" : "");
  const all = RB.matched === RB.got.length;
  $("rb-match").textContent = comma(RB.matched) + " of " + comma(RB.got.length); $("rb-match").style.color = all ? "var(--train)" : "#d03b3b";
  $("rb-dist").textContent = comma(r ? r.distinct : 0) + " · " + comma(r ? r.disagree : 0) + " wrong";
  $("rb-title").textContent = "Run it on the network · " + (it.src.title === "Fibonacci word fractal" ? "Fibonacci word" : it.src.title) + " · " + RUN.who;
  rbButtons();
}
$("rb-play").onclick = () => RB.playing ? rbStop() : rbPlay();
$("rb-step").onclick = () => { rbStop(); if (RB.run && RB.run.verdict) rbReset(); rbStep(); rbRender(); };
$("rb-watch").onclick = () => { RB.fast = false; rbButtons(); };
$("rb-fast").onclick = () => { RB.fast = true; rbButtons(); };
$("p-run").onclick = (ev) => { ev.preventDefault(); $("p-runbox").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "center" }); if (!RB.playing) rbPlay(); };
function rbChoose(it) { rbStop(); RB.it = it; rbField(it); rbReset(); }

// the gallery: each program's drawing from a direct run of its rules, with its default input; press one to run it
{
  const g = $("gallery");
  LIST.forEach(([, items]) => items.forEach((it) => {
    const sp = RV.SPECS[it.id], param = sp.param.value, m = RV.setup(it.id, it, param), want = RV.direct(m.start, m.rules, m.tape);
    const d = sp.draw(want, param, true, want.length, sp.finalOf ? sp.finalOf(want) : null);
    const b = document.createElement("button"); b.type = "button"; b.className = "tile";
    const label = sp.param.options ? sp.param.options.find(([v]) => v === param)[1] : param;
    b.innerHTML = "<span class='tt'><b>" + esc(it.src.title === "Fibonacci word fractal" ? "Fibonacci word" : it.src.title) + "</b><span class='tiny'>" + esc(sp.param.label + " " + label) + "</span></span><span class='ts'>" + d.svg + "</span><span class='muted'>" + esc(d.sub.split(" · ").pop()) + "</span>";
    b.onclick = () => { choose(it); $("p-runbox").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "center" }); rbPlay(); };
    g.appendChild(b);
  }));
}

// what a step costs: every program measured on the network (program-stats.js), the sieve to 1,680 last
const ST = window.TCT_STATS, short = (n) => n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + " billion" : n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + " million" : comma(n);
function costTable() {
  const rows = [...ST.rows, Object.assign({ big: true }, ST.big)], pp = (r) => r.passes / r.steps, lo = Math.log10(Math.min(...rows.map(pp))), hi = Math.log10(Math.max(...rows.map(pp)));
  const item = (id) => LIST.flatMap(([, items]) => items).find((x) => x.id === id);
  $("costs").innerHTML = "<tr><th>Program</th><th>States</th><th>Ring</th><th>Steps</th><th>Universal steps<br>per step</th><th style='width:34%'>Forward passes per step</th><th>The delta decoder<br>computes</th></tr>" +
    rows.map((r) => {
      const it = item(r.id);
      const w = 4 + 96 * (Math.log10(pp(r)) - lo) / (hi - lo || 1);
      return "<tr data-id='" + r.id + "'" + (r.big ? " class='big'" : RUN.it.id === r.id ? " class='on'" : "") + "><td><b>" + esc(progTitle(it)) + "</b><small>" + esc(RVZ.show(r.id, r.param)) + (r.big ? " · the sieve section" : "") + "</small></td>" +
        "<td>" + (r.id === "bb4" ? (r.states - 1) + " + 1" : comma(r.states)) + "</td><td>" + comma(r.ring) + "</td><td>" + comma(r.steps) + "</td><td>" + comma(Math.round(r.utmSteps / r.steps)) + "</td>" +
        "<td><span class='cbar'><i style='width:" + w.toFixed(1) + "%'></i></span><span class='cnum'>" + short(Math.round(pp(r))) + "</span></td>" +
        "<td>1 in " + comma(Math.round(r.passes / r.read)) + "</td></tr>";
    }).join("");
  document.querySelectorAll("#costs tr[data-id]").forEach((tr) => { if (!tr.classList.contains("big")) tr.onclick = () => { choose(item(tr.dataset.id)); $("v-hero").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "center" }); }; });
  const A = ST.all, B = ST.big;
  $("cost-note").textContent = "Ring: tokens in one ring at the start, the universal machine's whole tape. The busy beaver has one more state, a prelude that makes room on the one-way tape. Passes per step are forward passes the full decoder makes; the bar is on a log scale. " +
    "Every run here matched a direct run of its rules, configuration by configuration, with " + A.wrong + " wrong windows; " + comma(A.unseen + B.unseen) + " of the windows met were never in training.";
  const r = ST.rows.find((x) => x.id === RUN.it.id);
  $("tl-ring").textContent = comma(r.ring); $("tl-ring-l").textContent = "tokens in one ring of " + RUN.label.replace(" · ", ", ") + ", at the start";
  $("tl-rings").textContent = short(r.utmSteps); $("tl-rings-l").textContent = "rings written for that whole run, one per step of the universal machine";
  const min = ST.rows.reduce((a, x) => (x.passes / x.steps < a.passes / a.steps ? x : a));
  const lo1 = min.passes / min.steps, hi1 = B.passes / B.steps, minD = ST.rows.reduce((a, x) => Math.min(a, x.passes / x.read), Infinity);
  $("st-cost").textContent = comma(Math.round(hi1 / lo1 / 100) * 100) + "×";
  $("st-cost-l").textContent = "the forward passes per step, from the " + min.states + "-state " + progTitle(item(min.id)).toLowerCase() + " (" + comma(Math.round(lo1)) + ") to the 250-state sieve (" + short(Math.round(hi1)) +
    "): the same network, a longer ring. Memory is free; time is not. A delta decoder reads only the windows that changed: from 1 in " + comma(Math.round(minD)) + " to 1 in " + comma(Math.round(B.passes / B.read)) + ".";
}
// one current program for the page: picking it here, in the gallery, or in the scene's menu sets all three
function setHeroProgram(it) {
  $("h-prog").value = it.id;
  if (RUN.it.id === it.id) { RUN.it = it; costTable(); return; }
  setPlaying(false); heroProgram(it); newRun(); hero.anim = 1;
  if (heroMode === "run") render();
  costTable();
}
LIST.forEach(([grp, items]) => { const g = document.createElement("optgroup"); g.label = grp; items.forEach((it) => { const o = document.createElement("option"); o.value = it.id; o.textContent = progTitle(it); g.appendChild(o); }); $("h-prog").appendChild(g); });
$("h-prog").onchange = () => { const it = LIST.flatMap(([, items]) => items).find((x) => x.id === $("h-prog").value); choose(it); };
choose(LIST[0][1].find((it) => it.id === "collatz"));

// ---------------------------------------------------------------- the sieve
{
  const nums = []; for (let n = 2; n <= 31; n++) nums.push(n);
  let crossed = new Set(), html = "";
  [["START", 0], ["STRIDE 2", 2], ["STRIDE 3", 3], ["STRIDE 5", 5]].forEach(([name, k]) => {
    const fresh = new Set(nums.filter((n) => k && n > k && n % k === 0 && !crossed.has(n)));
    html += "<div class='row'><span>" + name + "</span>" + nums.map((n) => "<i class='" + (n === k ? "p" : fresh.has(n) ? "new" : crossed.has(n) ? "old" : "") + "'>" + n + "</i>").join("") + "</div>";
    fresh.forEach((n) => crossed.add(n));
  });
  $("sv-rows").innerHTML = html;
  const m = TCT.parseMachine(SIEVE.start, SIEVE.rules), st = m.order;
  const cnt = (re) => st.filter((q) => re.test(q)).length;
  const top = Math.floor(Math.sqrt(1680)), base = window.TCT_PRIMES.primesUpto(top);
  $("sv-states").innerHTML = [["Counting chains, one per prime from 2 to " + base[base.length - 1], cnt(/^m\d+_/)], ["“Is k prime?”, for k from 2 to " + top, cnt(/^k\d+$/)],
    ["Walking back, one per prime", cnt(/^r\d+$/)], ["Getting started", cnt(/^s1?$/)]].map(([a, b]) => "<div><span>" + a + "</span><span>" + b + "</span></div>").join("") +
    "<div><b>States</b><b class='mono' style='color:var(--accent)'>" + st.length + "</b></div>";
  const r3 = SIEVE.rules.filter((r) => /^(k3|m3_\d|r3) /.test(r));
  $("sv-count").textContent = r3.length + " OF " + SIEVE.rules.length + " RULES";
  const NOTE = { "k3 0": "crossed out: try 4", "k3 1": "prime: mark it, start counting", "m3_0 1": "every third cell: write 0", "m3_2 _": "end of the tape: turn back", "r3 _": "restore the mark, try 4" };
  const fmt = (r) => { const k = r.split(" ").slice(0, 2).join(" "); return NOTE[k] ? r.padEnd(22) + NOTE[k] : r; };
  const grp = (t, rs) => "<div class='code-group'><span class='tiny' style='color:var(--accent)'>" + t + "</span><pre>" + esc(rs.map(fmt).join("\n")) + "</pre></div>";
  const order = (a) => a.sort((x, y) => { const k = (r) => [r.startsWith("m3_2") ? 0 : r.startsWith("m3_1") ? 1 : 2, r.includes(" _ ") ? 1 : 0]; const [a1, b1] = k(x), [a2, b2] = k(y); return b1 - b2 || a1 - a2; });
  $("sv-rules").innerHTML = grp("Is 3 prime? (cell 3 still 1)", r3.filter((r) => r.startsWith("k3"))) +
    grp("Count 2, 1, 0 · cross out at 0", order(r3.filter((r) => r.startsWith("m3_")))) + grp("Walk back to the mark", r3.filter((r) => r.startsWith("r3")));
}
// the Ulam spiral, primes raised; they rise in order of n when the view comes into sight
{
  const v = addView("v-spiral", { pad: 1.02 }), N = 1680, pos = [null, [0, 0]];
  let x = 0, y = 0, n = 1, step = 1, di = 0; const dir = [[1, 0], [0, -1], [-1, 0], [0, 1]];
  while (n < N) { for (let rep = 0; rep < 2; rep++) { for (let s = 0; s < step && n < N; s++) { x += dir[di][0]; y += dir[di][1]; n++; pos[n] = [x, y]; } di = (di + 1) % 4; } step++; }
  const primes = window.TCT_PRIMES.primesUpto(N);
  v.root.add(plate(-20.6, -20.6, 20.6, 20.6, 0, C("#E3E8EB")));
  const m = barMesh(primes.length); v.root.add(m);
  const one = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.06, 0.9), new THREE.MeshBasicMaterial({ color: ORANGE })); one.position.y = 0.03; v.root.add(one);
  fit(v);
  let t0 = null;
  const draw = (p) => { primes.forEach((q, i) => { const k = Math.min(1, Math.max(0, p * N / q - 1) * 4 + (p * N >= q ? 0.0001 : 0)); setBar(m, i, pos[q][0], pos[q][1], 0.9 * Math.min(1, k), 0.72, NAVY); }); done(m); };
  draw(reduced ? 1 : 0);
  v.onVisible = (now) => { t0 = now / 1000; };
  v.tick = (t) => { if (t0 === null || reduced) return; const p = Math.min(1.25, (t - t0) / 5); draw(p); if (p >= 1.25) v.tick = null; };
  hoverable(v, () => [m], (mesh, i) => primes[i] + " is prime");
}

// ---------------------------------------------------------------- how it differs: three small scenes
{ // built by hand: a few layers, a few weights set to exact values, everything else exactly 0
  const v = addView("v-built", { pad: 1.08 });
  for (let L = 0; L < 4; L++) {
    const dx = L * 1.2, z = L * 5.5, g = new THREE.Group(); g.position.set(dx, z, -dx);
    g.add(plate(-0.5, -0.5, 9.5, 9.5, 0, C("#D3DADE"), 0.6));
    const bars = [];
    for (let i = 0; i < 10; i++) for (let j = 0; j < 10; j++) {
      const val = j === (3 * i + 2 * L) % 10 && i % 2 === 0 ? 1 : j === (7 * i + L) % 10 && i % 3 === 1 ? -1 : 0;
      if (val) bars.push([j, i, val]);
    }
    const m = barMesh(bars.length); bars.forEach(([j, i, val], k) => setBar(m, k, j, i, 1.4, 0.8, wColor(val, 1, tmpC))); done(m); g.add(m);
    v.root.add(g);
  }
  fit(v);
}
{ // found in a giant model: a vast dense field, and a long prompt in front of it, generated tokens appearing at its end
  const v = addView("v-giant", { pad: 1.05 }), r = rng(11), n = 46;
  const m = barMesh(n * n); let k = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const val = gauss(r); setBar(m, k++, j, i, Math.min(1, Math.abs(val) / 2.6) * 3, 0.9, wColor(val, 2.6, tmpC)); }
  done(m); v.root.add(m);
  const strip = barMesh(64); v.root.add(strip);
  const paint = (shift) => { for (let q = 0; q < 64; q++) { const gen = q >= 64 - 3 - shift; setBar(strip, q, -6 + q * 0.85, 52, 0.15, 0.7, gen ? ORANGE : q > 54 - shift ? SLATE : PALE); } done(strip); };
  paint(0); fit(v);
  let acc = 0, sh = 0; v.tick = (t, dt) => { if (reduced) return; acc += dt; if (acc > 0.7) { acc = 0; sh = (sh + 1) % 6; paint(sh); } };
}
{ // learned from steps: the real network, small, reading three tokens of a long transcript
  const v = addView("v-learned", { pad: 1.06 });
  const T_ = (M) => M[0].map((_, j) => M.map((row) => row[j]));
  const blocks = [[W["Wq.weight"], 0, -9], [W["Wk.weight"], 0, 0], [W["Wv.weight"], 0, 9], [W["Wo.weight"], 12, 0], [T_(W["W1.weight"]), 24, -12]];
  const cells = []; blocks.forEach(([M, bx, by]) => { const norm = Math.max(...M.flat().map(Math.abs)); M.forEach((row, i) => row.forEach((val, j) => cells.push([bx + j, by + i, val, norm]))); });
  const m = barMesh(cells.length); cells.forEach(([x, z, val, norm], k) => setBar(m, k, x, z, Math.max(0.05, Math.abs(val) / norm * 2.2), 0.8, wColor(val, norm, tmpC))); done(m); v.root.add(m);
  const tape = barMesh(46); v.root.add(tape);
  const paint = (w) => { for (let q = 0; q < 46; q++) setBar(tape, q, 2 + q * 1.25, 21.7, 0.15, 1, q >= w && q < w + 3 ? ORANGE : C("#D8DEE2")); done(tape); };
  paint(21); fit(v);
  let acc = 0, w = 6; v.tick = (t, dt) => { if (reduced) return; acc += dt; if (acc > 0.35) { acc = 0; w = w >= 40 ? 6 : w + 1; paint(w); } };
}

// ---------------------------------------------------------------- UMT: where the memory is
{ // the usual picture: a big model attends to every cell of a window; cells that leave it are gone
  const v = addView("v-context", { pad: 1.06 }), r = rng(3);
  const m = barMesh(16 * 24); let k = 0;
  for (let i = 0; i < 16; i++) for (let j = 0; j < 24; j++) { const val = gauss(r); setBar(m, k++, j + 2, i - 22, Math.min(1, Math.abs(val) / 2.6) * 2.4, 0.88, wColor(val, 2.6, tmpC)); }
  done(m); v.root.add(m);
  const nIn = 30, tape = barMesh(nIn + 12); v.root.add(tape);
  const win = new THREE.Mesh(new THREE.BoxGeometry(nIn + 0.2, 1.2, 1.4), new THREE.MeshBasicMaterial({ color: NAVY, transparent: true, opacity: 0.12, depthWrite: false }));
  win.position.set(nIn / 2 - 0.5, 0.6, 2); v.root.add(win);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(win.geometry), new THREE.LineBasicMaterial({ color: NAVY })); edges.position.copy(win.position); v.root.add(edges);
  const lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: NAVY, transparent: true, opacity: 0.18 })); v.root.add(lines);
  const top = new THREE.Vector3(14, 8, -14), seq = Array.from({ length: 400 }, () => r() < 0.35 ? 1 : 0);
  const paint = (off) => {
    const pts = [];
    for (let q = 0; q < nIn + 12; q++) {
      const val = seq[(q + off) % seq.length], inside = q < nIn;
      tmpC.copy(val ? SLATE : C("#C4CED5")); if (!inside) tmpC.lerp(WHITE, 0.72);
      setBar(tape, q, q, 2, 0.3, 0.84, tmpC);
      if (inside) pts.push(top.x, top.y, top.z, q, 0.32, 2);
    }
    done(tape); lines.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  };
  paint(0); fit(v);
  let acc = 0, off = 0; v.tick = (t, dt) => { if (reduced) return; acc += dt; if (acc > 0.5) { acc = 0; off++; paint(off); } };
}
{ // UMT: the transcript, ring after ring; the tiny network reads three tokens of the newest ring and writes the next
  const v = addView("v-transcript", { pad: 1.05 }), R = 9, L = 30;
  const cells = barMesh(R * L + L), rr = rng(9); v.root.add(cells);
  const content = Array.from({ length: R + 1 }, () => Array.from({ length: L }, () => (rr() < 0.4 ? 1 : 0)));
  const M = W["Wq.weight"], norm = Math.max(...M.flat().map(Math.abs));
  const net = barMesh(64); v.root.add(net);
  const lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: ORANGE })); v.root.add(lines);
  let w0 = 0;
  const paint = () => {
    let k = 0;
    for (let r = 0; r < R; r++) {                        // r = R - 1 is the newest complete ring, nearest the viewer
      const z = -(R - 1 - r) * 2.2, fade = 0.3 + 0.7 * r / (R - 1), head = 6 + r;
      for (let q = 0; q < L; q++) {
        const isWin = r === R - 1 && q >= w0 && q < w0 + 3;
        tmpC.copy(isWin ? ORANGE : q === head ? ORANGE : content[r][q] ? SLATE : C("#C4CED5"));
        if (!isWin) tmpC.lerp(WHITE, (1 - fade) * 0.85 + (q === head ? 0.25 : 0));
        setBar(cells, k++, q, z, isWin ? 0.32 : 0.14, 0.84, tmpC);
      }
    }
    for (let q = 0; q < L; q++) {                        // the ring being written: up to the window, then empty
      const written = q <= w0 + 1;
      tmpC.copy(q === w0 + 1 ? ORANGE : content[R][q] ? SLATE : C("#C4CED5"));
      setBar(cells, k++, q, 2.2, written ? (q === w0 + 1 ? 0.32 : 0.14) : 0.002, 0.84, tmpC);
    }
    done(cells);
    let i = 0; M.forEach((row, a) => row.forEach((val, b) => setBar(net, i++, w0 - 1.2 + b * 0.62, -4.2 + a * 0.62, Math.max(0.05, Math.abs(val) / norm), 0.42, wColor(val, norm, tmpC))));
    done(net);
    const hub = [w0 + 1, 4.6, -2], pts = [];
    for (let q = w0; q < w0 + 3; q++) pts.push(...hub, q, 0.32, 0);
    pts.push(...hub, w0 + 1, 0.32, 2.2);
    lines.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  };
  paint(); fit(v);
  let acc = 0; v.tick = (t, dt) => { if (reduced) return; acc += dt; if (acc > 0.32) { acc = 0; w0 = w0 >= L - 4 ? 0 : w0 + 1; paint(); } };
}

// ---------------------------------------------------------------- constraints: the busy beaver on a tape, and the same run through time
const BB2 = { "A_": ["1", "R", "B"], "A1": ["1", "L", "B"], "B_": ["1", "L", "A"], "B1": ["1", "R", "H"] };
const bbRun = (() => { const tape = {}, out = []; let pos = 0, st = "A"; out.push({ tape: { ...tape }, pos, st });
  while (st !== "H") { const [w, mv, n] = BB2[st + (tape[pos] || "_")]; tape[pos] = w; pos += mv === "R" ? 1 : -1; st = n; out.push({ tape: { ...tape }, pos, st }); } return out; })();
{
  const v = addView("v-tm", { pad: 1.1 }), L = 14, off = 7;
  const tape = barMesh(L); v.root.add(tape);
  const head = new THREE.Mesh(new THREE.BoxGeometry(1.16, 0.08, 1.16), new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: 0.6 })); v.root.add(head);
  const ctrl = new THREE.Mesh(new THREE.BoxGeometry(3.2, 1.8, 3.2), new THREE.MeshLambertMaterial({ color: NAVY })); v.root.add(ctrl);
  const lamp = new THREE.Mesh(new THREE.BoxGeometry(1, 0.05, 1), new THREE.MeshBasicMaterial({ color: ORANGE })); v.root.add(lamp);
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 3), new THREE.MeshBasicMaterial({ color: NAVY })); v.root.add(stem);
  const show = (i) => {
    const c = bbRun[i];
    for (let q = 0; q < L; q++) { const one = c.tape[q - off] === "1"; setBar(tape, q, q, 0, one ? 0.5 : 0.25, 0.88, one ? NAVY : C("#E2E7EA")); }
    done(tape);
    const x = c.pos + off; head.position.set(x, 0.3, 0); ctrl.position.set(x, 4.3, 0); lamp.position.set(x, 5.22, 0); stem.position.set(x, 1.9, 0);
    lamp.material.color.copy(c.st === "A" ? ORANGE : c.st === "B" ? C("#F7A27A") : SLATE);
  };
  show(0); fit(v);
  let acc = 0, i = 0; v.tick = (t, dt) => { if (reduced) return; acc += dt; if (acc > (i === bbRun.length - 1 ? 2.4 : 0.9)) { acc = 0; i = (i + 1) % bbRun.length; show(i); } };
}
{ // the same run, ring after ring down the page, with the light cones from three cells to one
  const v = addView("v-cones", { pad: 1.06 }), L = 14, off = 7, R = bbRun.length;
  const cells = barMesh(R * L); v.root.add(cells);
  let k = 0;
  bbRun.forEach((c, r) => { for (let q = 0; q < L; q++) { const one = c.tape[q - off] === "1", hd = c.pos + off === q;
    setBar(cells, k++, q, 0, 0.18, 0.84, hd ? ORANGE : one ? SLATE : C("#E2E7EA")); } });
  done(cells);
  // rings stacked downward in time: each ring a row of its own at its own height
  for (let r = 0; r < R; r++) for (let q = 0; q < L; q++) { cells.getMatrixAt(r * L + q, tmpM); tmpM.setPosition(q, -(r) * 2.0, 0); cells.setMatrixAt(r * L + q, tmpM); }
  done(cells);
  const cones = new THREE.Group(); v.root.add(cones);
  const coneMat = new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false });
  const strong = new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: 0.45, side: THREE.DoubleSide, depthWrite: false });
  const tri = (q, r, mat) => { const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute([q - 1, -r * 2.0, 0, q + 1, -r * 2.0, 0, q, -(r + 1) * 2.0 + 0.18, 0], 3)); return new THREE.Mesh(g, mat); };
  for (let r = 0; r < R - 1; r++) for (let q = 1; q < L - 1; q += 3) cones.add(tri(q, r, coneMat));
  const hot = new THREE.Group(); v.root.add(hot);
  fit(v);
  let acc = 0, r = 0;
  const showCone = () => { hot.clear(); const q = bbRun[r + 1].pos + off; hot.add(tri(q, r, strong)); };
  showCone();
  v.tick = (t, dt) => { if (reduced) return; acc += dt; if (acc > 0.9) { acc = 0; r = (r + 1) % (R - 1); showCone(); } };
}

// ---------------------------------------------------------------- many at once: every two-state machine
{
  const CH = []; for (const w of "_1") for (const mv of "LR") for (const n of "ABH") CH.push([w, mv, n]);
  const KEYS = ["A_", "A1", "B_", "B1"], N = 20736, steps = new Int8Array(N), ones = new Int8Array(N);
  const tableOf = (i) => { const c = [Math.floor(i / 1728) % 12, Math.floor(i / 144) % 12, Math.floor(i / 12) % 12, i % 12]; const t = {}; KEYS.forEach((k, j) => { t[k] = CH[c[j]]; }); return t; };
  for (let i = 0; i < N; i++) {                          // run each machine directly, 7 steps at most (BB(2) = 6)
    const t = tableOf(i), tape = {}; let pos = 0, st = "A", s = 0, halted = false;
    while (s < 7) { const [w, mv, n] = t[st + (tape[pos] || "_")]; tape[pos] = w; pos += mv === "R" ? 1 : -1; s++; st = n; if (st === "H") { halted = true; break; } }
    steps[i] = halted ? s : -1; ones[i] = Object.values(tape).filter((x) => x === "1").length;
  }
  const v = addView("v-field", { pad: 1.02 }), m = barMesh(N); v.root.add(m);
  const place = (i) => { const r = Math.floor(i / 144), c = i % 144; return [c + Math.floor(c / 8) * 0.7, r + Math.floor(r / 8) * 0.7]; };
  const isChamp = (i) => steps[i] === 6 && ones[i] === 4;
  const H = 1.25;
  const paint = (s) => {                                   // the field after s steps
    let halted = 0;
    for (let i = 0; i < N; i++) {
      const [x, z] = place(i), st = steps[i];
      if (st > 0 && st <= s) {
        halted++;
        tmpC.copy(isChamp(i) ? ORANGE : C("#B0BEC8").lerp(NAVY, 0.35 + 0.65 * st / 6));
        setBar(m, i, x, z, st * H, 0.8, tmpC);
      } else if (s >= 7) setBar(m, i, x, z, 0.06, 0.84, FOG);          // never halts: lies flat
      else setBar(m, i, x, z, Math.max(0.06, s * H), 0.8, C("#DCE3E7"));  // still running
    }
    done(m);
    $("f-halt").textContent = comma(halted);
    $("f-status").textContent = "Step " + s + " of 7 · each machine run by its own rules";
    if (s >= 7) { $("f-never").textContent = comma(N - halted); let mx = 0, mo = 0; for (let i = 0; i < N; i++) if (steps[i] > 0) { mx = Math.max(mx, steps[i]); mo = Math.max(mo, ones[i]); } $("f-rec").textContent = mx + " · " + mo; }
    else { $("f-never").textContent = "—"; $("f-rec").textContent = "—"; }
  };
  paint(0); fit(v);
  let s = 0, acc = 0, running = false;
  const start = () => { s = 0; acc = 0; running = true; paint(0); };
  $("f-run").onclick = start;
  v.onVisible = () => { if (!reduced) setTimeout(start, 400); else paint(7); };
  v.tick = (t, dt) => { if (!running) return; acc += dt; if (acc > 0.7) { acc = 0; s++; paint(s); if (s >= 7) running = false; } };
  hoverable(v, () => [m], (mesh, i) => {
    const t = tableOf(i), rules = KEYS.map((k) => k[0] + " " + k[1] + " -> " + (t[k][2] === "H" ? "halt" : t[k][2]) + " " + t[k][0] + " " + t[k][1]).join("\n");
    return "machine " + comma(i + 1) + " · " + (steps[i] > 0 ? "halts after " + steps[i] + " step" + (steps[i] > 1 ? "s" : "") + ", " + ones[i] + " ones" : "never halts") + "\n" + rules;
  });
}
window.__umt = { views };
