// tlf.html: the Universal Machine Transformer page. Every 3D view draws into its own slot of one shared WebGL canvas
// (a page can only hold a few WebGL contexts). The numbers and weights come from the code and the trained network.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { analyse, windows, INSIDE, engine, vin, vout, D, W, TCT } from "./inside-scene.js";

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
function networkScene(v, H = 2.6) {
  const m = barMesh(NET.cells.length); v.root.add(m); v.net = m; v.vals = new Float32Array(NET.cells.length);
  v.paint = (p, prev) => {
    const norm = {}, move = {};
    BLOCKNAMES.forEach((b) => { norm[b] = 1e-9; move[b] = 1e-9; });
    NET.cells.forEach((c) => { norm[c.block] = Math.max(norm[c.block], Math.abs(p[c.off])); if (prev) move[c.block] = Math.max(move[c.block], Math.abs(p[c.off] - prev[c.off])); });
    NET.cells.forEach((c, i) => {
      const val = p[c.off], n = norm[c.block]; v.vals[i] = val;
      wColor(val, n, tmpC);
      if (prev) tmpC.lerp(GLASS, 0.85 * Math.min(1, Math.abs(val - prev[c.off]) / move[c.block]));
      setBar(m, i, c.x, c.z, Math.max(0.05, Math.min(1, Math.abs(val) / n) * H), 0.8, tmpC);
    });
    done(m);
  };
  v.paint(SHIPPED);
  return m;
}

// ---------------------------------------------------------------- hero: the network, running and training for real
const hero = addView("v-hero", { pad: 1.04 });
networkScene(hero);
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

// Run: every pulse is a real forward pass of the network on the next window of the sieve to 120 that has the head in
// it, written into the transcript and checked against the universal machine's exact rule.
const SV = window.TCT_PRIMES.sieve(120), SVM = TCT.parseMachine(SV.start, SV.rules);
const runner = { engine, who: "the network on this page", run: null, checked: 0, wrong: 0 };
function newRun() { runner.run = new TCT.Run(runner.engine, SVM, SV.tape); runner.checked = 0; runner.wrong = 0; }
function chips(toks, out, note) {
  $("h-window").innerHTML = "<span class='tiny'>Reads</span>" + toks.map((t) => "<span class='chip" + (t.includes(":") ? " head" : "") + "'>" + esc(t) + "</span>").join("") +
    "<span class='muted' style='padding:0 6px'>→</span><span class='tiny'>Writes</span><span class='chip out'>" + esc(out || "(halt)") + "</span>" +
    "<span class='muted' style='font-size:14px; padding-left:8px'>" + note + "</span>";
}
function onePass() {
  const r = runner.run, e = runner.engine;
  let n = 0;
  do { r.step(false); n++; } while (!r.verdict && n < 20000 && !r.window().some((id) => vin[id].includes(":")));
  if (r.verdict) { newRun(); return onePass(); }
  const ids = r.window(), f = e.net.forward(ids), lg = f.logits;
  let mx = -Infinity; for (const x of lg) mx = Math.max(mx, x); let z = 0; for (const x of lg) z += Math.exp(x - mx);
  const wrote = D.emits[vout[f.best]].join(" "), exact = e.exact(ids), ok = wrote === exact;
  runner.checked++; if (!ok) runner.wrong++;
  chips(ids.map((i) => vin[i]), wrote, Math.round(Math.exp(lg[f.best] - mx) / z * 100) + "% sure · " + (e.trained.has(e.key(ids)) ? "in training" : "never in training") + " · " +
    (ok ? "right" : "<b style='color:#d03b3b'>wrong</b>") + " · " + comma(runner.checked) + " checked, " + runner.wrong + " wrong");
  $("h-status").textContent = "Forward pass " + comma(r.passes + 1) + " of the sieve to 120 · " + runner.who;
}
newRun();

// Train: real full-batch Adam on the GPU (train-gpu.js), from the trainer's random start, on all 100,697 training steps,
// learning rate 0.003, until every training step is right: the recipe that trained the network on this page.
const trainer = { gpu: null, state: "idle", step: 0, t0: 0, n: 0, prev: null, caught: false, latest: null };
function trainNote(html) { const el = $("h-train-note"); el.innerHTML = html; el.hidden = false; }
async function startTraining() {
  if (trainer.state === "running") return;
  if (trainer.state === "paused") { trainer.state = "running"; trainLoop(); return; }
  if (!navigator.gpu) { trainNote("Training runs on WebGPU, which this browser does not offer. Chrome or Edge on a recent computer will. The network on this page was trained the same way."); return; }
  trainer.state = "starting";
  trainNote("Starting the GPU, and loading the 100,697 training steps…");
  try {
    if (!trainer.gpu) {
      trainer.gpu = await window.TrainGPU.create(LAY, { fusedHead: true });
      const ids = D.trainIds, n = ids.length / 3, labels = new Uint32Array(n), index = new Map(vout.map((v, i) => [v, i]));
      for (let k = 0; k < n; k++) { const ex = engine.exact([ids[3 * k], ids[3 * k + 1], ids[3 * k + 2]]); labels[k] = index.get(ex === "" ? "<halt>" : ex); }
      trainer.gpu.setData({ steps: ids, labels }, Array.from({ length: n }, (_, i) => i));
      trainer.n = n;
    }
    trainer.gpu.setParams(INIT); trainer.step = 0; trainer.prev = INIT; trainer.caught = false; trainer.t0 = performance.now();
    hero.paint(INIT);
    trainer.state = "running"; trainNote("Training on your GPU: full-batch Adam on all 100,697 training steps…"); trainLoop();
  } catch (err) { trainer.state = "idle"; trainNote("The GPU could not start training here (" + esc(err.message || err) + "). The network on this page was trained the same way."); }
}
async function trainLoop() {
  const g = trainer.gpu;
  while (trainer.state === "running") {
    const r = await g.step(0.003); trainer.step++;
    const frac = r.correct / trainer.n;
    if (trainer.step % 3 === 0 || r.correct === trainer.n) {
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
      runner.engine = new TCT.Engine(Object.assign({}, D, { weights: toWeights(p) })); runner.who = "your network, trained just now"; newRun();
      trainNote("Every training step right at Adam step " + comma(trainer.step) + ", after " + secs.toFixed(0) + " seconds on your GPU. Switch to <b>Run</b>: the pass now runs the network you just trained, checked against the exact rule.");
    } else if (trainer.step >= 8000) { trainer.state = "done"; trainNote("Stopped at Adam step 8,000 with " + comma(r.correct) + " of " + comma(trainer.n) + " training steps right."); }
  }
}

let heroMode = "run", playing = !reduced, heroT = 0, lastCycle = -1;
const PERIOD = 2.6;
hero.tick = (t, dt) => {
  if (heroMode !== "run") return;
  if (playing) heroT += dt;
  const cyc = Math.floor(heroT / PERIOD), u = (heroT / PERIOD) % 1;
  if (cyc !== lastCycle) { lastCycle = cyc; onePass(); }
  passAt(u, pulse.position);
  $("h-fill").style.width = (u * 100).toFixed(1) + "%";
};
function setMode(m) {
  heroMode = m;
  $("mode-run").setAttribute("aria-pressed", m === "run"); $("mode-train").setAttribute("aria-pressed", m === "train");
  [passTube, pulse, ...dots, ...dots.map((d) => d.userData.ring).filter(Boolean)].forEach((o) => { o.visible = m === "run"; });
  $("h-play").setAttribute("aria-label", m === "train" ? "Train" : playing ? "Pause" : "Play");
  if (m === "run") {
    $("h-train-note").hidden = trainer.state !== "done";
    hero.paint(runner.who.startsWith("your") ? trainer.latest : SHIPPED); lastCycle = -1;
  } else {
    if (trainer.latest) hero.paint(trainer.latest); else hero.paint(INIT);
    if (trainer.state === "idle") {
      $("h-status").textContent = "A fresh network, from a random start · press ▶ to train";
      $("h-window").innerHTML = "<span class='muted' style='font-size:14px'>Full-batch Adam on your GPU, on all 100,697 training steps, until every one is right.</span>";
      $("h-fill").style.width = "0%";
      trainNote("These are the trainer's random starting weights. Press ▶ and gradient descent trains them, here, on your GPU; bars glow sea glass where the weights are moving.");
    }
  }
}
$("mode-run").onclick = () => setMode("run"); $("mode-train").onclick = () => setMode("train");
$("h-play").onclick = () => {
  if (heroMode === "train") { if (trainer.state === "running") { trainer.state = "paused"; trainNote("Paused at Adam step " + comma(trainer.step) + ". Press ▶ to go on."); } else startTraining(); return; }
  playing = !playing; $("h-play").setAttribute("aria-label", playing ? "Pause" : "Play");
};
hoverable(hero, () => [hero.net], (m, i) => { const c = NET.cells[i]; return c.name + "[" + c.i + "][" + c.j + "] = " + hero.vals[i].toFixed(3); });
// the block labels, placed under the blocks wherever the camera puts them
function placeLabels() {
  const el = $("h-labels"); if (!hero.aspect) return requestAnimationFrame(placeLabels);
  el.innerHTML = "";
  const groups = [0, 1, 2, 3, 4, 5].map((g) => NET.centers.filter((c) => c.group === g));
  groups.forEach((cs, g) => {
    const p = cs[0].p.clone(); if (cs.length > 1) p.copy(cs[1].p);
    p.project(hero.camera);
    const s = document.createElement("span"); s.style.left = ((p.x + 1) / 2 * 100) + "%";
    s.innerHTML = "<b" + (g === 5 ? " style='color:var(--accent)'" : "") + ">" + LABELS[g][0] + "</b><small>" + LABELS[g][1] + "</small>";
    el.appendChild(s);
  });
}
placeLabels(); addEventListener("resize", () => { hero.aspect = 0; requestAnimationFrame(() => requestAnimationFrame(placeLabels)); });

// the Run column: the first input of the sieve the network never saw in training, the ring it reads and the ring it writes
const src = windows(), seen = analyse(src.unseen(0));
{
  const u0 = INSIDE.unseen[0], u1 = INSIDE.unseen[1];
  const tok = (t, cls) => "<span class='tok " + (cls || "") + "'>" + esc(t) + "</span>";
  const read = u0.before.slice(-4).map((t) => tok(t)).join("") + "<span class='winbox'>" + tok(u0.window[0], "hd") + tok(u0.window[1], "w") + tok(u0.window[2], "w") + "</span>" + u0.after.slice(0, 2).map((t) => tok(t)).join("");
  const write = u0.before.slice(-4).map((t) => tok(t)).join("") + tok(u1.window[0]) + tok(seen.wrote, "out") + tok("?", "q");
  $("run-rings").innerHTML = "<div class='ring-row'><span class='tiny'>Ring " + comma(u0.utmStep) + "</span>" + read + "</div><div class='ring-row'><span class='tiny'>Ring " + comma(u0.utmStep + 1) + "</span>" + write + "</div>";
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
    { id: "counter", src: prog("counter"), input: prog("counter").input(3), shown: "3 bits, all 0", run: "programs.html?p=counter&utm=2", stats: ["8 bits", 1014, 9417] },
    { id: "sort", src: prog("sort"), input: prog("sort").input("1011"), shown: "1011", run: "programs.html?p=sort&utm=2", stats: ["16 digits", 648, 26534] },
    { id: "count", src: prog("count"), input: prog("count").input(3), shown: "three 1s", run: "programs.html?p=count&utm=2", stats: ["40 ones", 3478, 60472] },
    { id: "add", src: prog("add"), input: prog("add").input("3+2"), shown: "3 + 2", run: "programs.html?p=add&utm=2", stats: ["1000 + 99", 1999, 58231] },
    { id: "collatz", src: prog("collatz"), input: prog("collatz").input(6), shown: "n = 6", run: "programs.html?p=collatz&utm=2", stats: ["n = 27", 882, 81928] },
    { id: "fib", src: prog("fib"), input: prog("fib").input(4), shown: "4 bits", run: "programs.html?p=fib&utm=2", stats: ["8 bits", 634, 224735] }]],
  ["Drawings", [
    { id: "dragon", src: curve("dragon"), input: "_000", shown: "3 bits", run: "curves.html?c=dragon", stats: ["8 bits", 2528, 56762] },
    { id: "koch", src: curve("koch"), input: "_000", shown: "3 bits", run: "curves.html?c=koch", stats: ["10 bits", 1014, 85086] },
    { id: "fibword", src: curve("fibword"), input: "_000", shown: "3 bits", run: "curves.html?c=fibword", stats: ["8 bits", 492, 64868] }]],
  ["Records and scale", [
    { id: "bb4", src: { title: "Busy beaver", start: "A", rules: BB4, blurb: "The four-state busy beaver: the four-state machine that runs longest before stopping, 107 steps, leaving 13 ones." }, input: "_", shown: "a blank tape", run: "beaver.html", stats: null },
    { id: "sieve", src: { title: "Prime sieve", start: "s", rules: SIEVE.rules, blurb: "The sieve of Eratosthenes up to 1,680, explained in full in the next section." }, input: SIEVE.tape, shown: "1,681 cells", run: "#sieve", stats: ["to 1,680", 39991, 246590379], big: true }]],
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
  $("p-run").href = it.run;
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
  $("p-runrow").innerHTML = it.stats ? "<span><span class='label accent'>3</span> <span class='label'>Run it · " + esc(it.stats[0]) + "</span></span>" +
    "<div><span class='num'>" + comma(it.stats[1]) + "</span><div class='muted' style='font-size:13px'>steps of the program</div></div>" +
    "<div><span class='num'>" + comma(it.stats[2]) + "</span><div class='muted' style='font-size:13px'>forward passes per step</div></div>" +
    "<div><span class='num' style='color:var(--accent)'>all right</span><div class='muted' style='font-size:13px'>checked against the rules, step by step</div></div>" :
    "<span><span class='label accent'>3</span> <span class='label'>Run it</span></span><span class='muted'>On the busy beaver page, with every step checked against the rules.</span>";
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
choose(LIST[0][1][0]);

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

// ---------------------------------------------------------------- TLF: where the memory is
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
{ // TLF: the transcript, ring after ring; the tiny network reads three tokens of the newest ring and writes the next
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
    $("f-status").textContent = "Step " + s + " of 7 · one machine per GPU thread · blocks of 64";
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
window.__tlf = { views };
