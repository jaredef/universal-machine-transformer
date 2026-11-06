// The training page's 3D views (three.js, loaded from a CDN): the whole step table as a stack of cubes, and the
// token embeddings by their three principal directions. Both accept updates before three.js has loaded and apply the
// latest once it has.
(function (root) {
  "use strict";
  var THREE = null, Orbit = null, waiting = [];
  var ready = Promise.all([import("three"), import("three/addons/controls/OrbitControls.js")]).then(function (m) {
    THREE = m[0]; Orbit = m[1].OrbitControls; waiting.forEach(function (f) { f(); }); waiting = [];
  });
  function whenReady(f) { if (THREE) f(); else waiting.push(f); }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function reduced() { return matchMedia("(prefers-reduced-motion: reduce)").matches; }

  function Stage(el, distance) {
    var self = this;
    this.el = el; el.innerHTML = "";
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    this.camera.position.set(distance * 0.75, distance * 0.45, distance * 0.75);
    this.controls = new Orbit(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.autoRotate = !reduced(); this.controls.autoRotateSpeed = 0.6;
    // zoom only with a pinch or ⌘/Ctrl + scroll, so scrolling the page past a view never zooms it away, and never so far
    // that the scene leaves the camera's range (its far plane is 100)
    this.controls.enableZoom = false; this.controls.minDistance = distance * 0.3; this.controls.maxDistance = distance * 3;
    el.addEventListener("wheel", function (e) { self.controls.enableZoom = e.ctrlKey || e.metaKey; }, { capture: true, passive: true });
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    var sun = new THREE.DirectionalLight(0xffffff, 1.1); sun.position.set(3, 5, 2); this.scene.add(sun);
    this.resize = function () {
      var w = el.clientWidth, h = el.clientHeight;
      self.renderer.setSize(w, h, false); self.camera.aspect = w / h; self.camera.updateProjectionMatrix();
    };
    new ResizeObserver(this.resize).observe(el); this.resize();
    this.visible = true;
    new IntersectionObserver(function (e) { self.visible = e[0].isIntersecting; }).observe(el);
    this.hooks = [];
    (function loop() {
      requestAnimationFrame(loop);
      if (!self.visible) return;
      self.controls.update();
      self.hooks.forEach(function (f) { f(); });
      self.renderer.render(self.scene, self.camera);
    })();
  }

  // ---------------------------------------------------------------- the step table
  function TableView(el, table) {
    this.el = el; this.table = table; this.state = null; this.filter = null; var self = this;
    var n = table.labels.length;
    // stack the steps in the order the random machines first produced them (never-produced last)
    var order = Array.from({ length: n }, function (_, i) { return i; });
    order.sort(function (a, b) { var fa = table.first[a] < 0 ? 1e9 : table.first[a], fb = table.first[b] < 0 ? 1e9 : table.first[b]; return fa - fb || a - b; });
    this.slot = new Int32Array(n); order.forEach(function (k, i) { self.slot[k] = i; });
    this.side = Math.ceil(Math.cbrt(n));
    whenReady(function () { self.build(); });
  }
  TableView.prototype.build = function () {
    var self = this, n = this.table.labels.length, s = this.side;
    this.stage = new Stage(this.el, 3.2);
    var geo = new THREE.BoxGeometry(0.82, 0.82, 0.82), mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    this.mesh = new THREE.InstancedMesh(geo, mat, n);
    var m = new THREE.Matrix4(), scale = 1.6 / s;
    for (var k = 0; k < n; k++) {
      var i = this.slot[k], x = i % s, z = Math.floor(i / s) % s, y = Math.floor(i / (s * s));
      m.makeScale(scale, scale, scale); m.setPosition((x - s / 2) * scale, (y - s / 2) * scale, (z - s / 2) * scale);
      this.mesh.setMatrixAt(k, m);
    }
    this.stage.scene.add(this.mesh);
    this.colors = { learned: new THREE.Color(), pending: new THREE.Color(), good: new THREE.Color("#0ca30c"), bad: new THREE.Color("#d03b3b"), off: new THREE.Color() };
    this.theme(); this.apply();
  };
  TableView.prototype.theme = function () {
    if (!this.colors) return;
    this.colors.learned.set(css("--learned") || css("--accent") || "#2a78d6");
    this.colors.pending.set(css("--rule") || "#cccccc");
    this.colors.off.set(css("--soft") || "#eeeeee");
  };
  // inTrain: Uint8Array per step; pred: Int32Array per step, or null before the first evaluation
  TableView.prototype.update = function (inTrain, pred) { this.state = { inTrain: inTrain, pred: pred }; if (this.mesh) this.apply(); };
  TableView.prototype.setFilter = function (steps) { this.filter = steps ? new Set(steps) : null; if (this.mesh) this.apply(); };
  TableView.prototype.apply = function () {
    if (!this.state) return;
    var st = this.state, lab = this.table.labels, c = this.colors, m = new THREE.Matrix4(), self = this;
    var scale = 1.6 / this.side, s = this.side;
    for (var k = 0; k < lab.length; k++) {
      var shown = !this.filter || this.filter.has(k), col;
      if (!st.pred) col = st.inTrain[k] ? c.pending : c.off;
      else if (st.inTrain[k]) col = st.pred[k] === lab[k] ? c.learned : c.pending;
      else col = st.pred[k] === lab[k] ? c.good : c.bad;
      this.mesh.setColorAt(k, col);
      var i = this.slot[k], x = i % s, z = Math.floor(i / s) % s, y = Math.floor(i / (s * s)), sc = shown ? scale : 0;
      m.makeScale(sc, sc, sc); m.setPosition((x - s / 2) * scale, (y - s / 2) * scale, (z - s / 2) * scale);
      this.mesh.setMatrixAt(k, m);
    }
    this.mesh.instanceColor.needsUpdate = true; this.mesh.instanceMatrix.needsUpdate = true;
  };

  // ---------------------------------------------------------------- the embeddings
  function EmbView(el, vin, marked) {
    this.el = el; this.vin = vin; this.marked = marked; this.target = null; var self = this;
    this.cat = vin.map(function (t) { return t.indexOf(":") < 0 ? 0 : /^go_/.test(t) ? 2 : 1; });
    this.prevAxes = null;
    whenReady(function () { self.build(); });
  }
  EmbView.prototype.build = function () {
    var self = this, n = this.vin.length;
    this.stage = new Stage(this.el, 4.2);
    var geo = new THREE.SphereGeometry(0.035, 12, 8), mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    this.mesh = new THREE.InstancedMesh(geo, mat, n);
    this.stage.scene.add(this.mesh);
    this.pos = new Float32Array(n * 3); this.cur = new Float32Array(n * 3);
    this.lineGeo = new THREE.BufferGeometry(); this.lineGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    this.line = new THREE.Line(this.lineGeo, new THREE.LineBasicMaterial({ color: 0xffffff }));
    this.stage.scene.add(this.line);
    this.labels = this.marked.map(function (t) { var d = document.createElement("span"); d.className = "v-label"; d.textContent = t; self.el.appendChild(d); return d; });
    this.markIdx = this.marked.map(function (t) { return self.vin.indexOf(t); });
    this.theme();
    this.stage.hooks.push(function () { self.frame(); });
    if (this.target) this.place(this.target, true);
  };
  EmbView.prototype.theme = function () {
    if (!this.mesh) return;
    var dark = getComputedStyle(document.body).backgroundColor.match(/\d+/g).slice(0, 3).reduce(function (a, b) { return a + +b; }, 0) < 300;
    var pal = dark ? ["#3987e5", "#d95926", "#199e70"] : ["#2a78d6", "#eb6834", "#1baf7a"], c = new THREE.Color(), self = this;
    for (var i = 0; i < this.vin.length; i++) { c.set(pal[this.cat[i]]); this.mesh.setColorAt(i, c); }
    this.mesh.instanceColor.needsUpdate = true;
    this.line.material.color.set(css("--ink") || "#111111");
  };
  // E: the embedding matrix (nin x 64, row-major). Projects it on its top three principal directions.
  EmbView.prototype.update = function (E, width) {
    var n = this.vin.length, d = width || 64, mean = new Float64Array(d), i, j, k;
    for (i = 0; i < n; i++) for (j = 0; j < d; j++) mean[j] += E[i * d + j] / n;
    var X = new Float64Array(n * d);
    for (i = 0; i < n; i++) for (j = 0; j < d; j++) X[i * d + j] = E[i * d + j] - mean[j];
    var Cv = new Float64Array(d * d);
    for (i = 0; i < n; i++) for (j = 0; j < d; j++) { var xj = X[i * d + j]; if (!xj) continue; for (k = 0; k < d; k++) Cv[j * d + k] += xj * X[i * d + k]; }
    var axes = [], prev = this.prevAxes;
    for (var a = 0; a < 3; a++) {                            // power iteration with deflation, warm-started
      var v = new Float64Array(d);
      for (j = 0; j < d; j++) v[j] = prev ? prev[a][j] : Math.sin(7 * j + a);
      for (var it = 0; it < 60; it++) {
        var w = new Float64Array(d);
        for (j = 0; j < d; j++) { var s = 0; for (k = 0; k < d; k++) s += Cv[j * d + k] * v[k]; w[j] = s; }
        axes.forEach(function (u) { var dot = 0; for (var q = 0; q < d; q++) dot += w[q] * u[q]; for (q = 0; q < d; q++) w[q] -= dot * u[q]; });
        var norm = Math.sqrt(w.reduce(function (t, x) { return t + x * x; }, 0)) || 1;
        for (j = 0; j < d; j++) v[j] = w[j] / norm;
      }
      if (prev) { var sg = 0; for (j = 0; j < d; j++) sg += v[j] * prev[a][j]; if (sg < 0) for (j = 0; j < d; j++) v[j] = -v[j]; }
      axes.push(v);
    }
    this.prevAxes = axes;
    var P = new Float32Array(n * 3), maxAbs = 1e-9;
    for (i = 0; i < n; i++) for (a = 0; a < 3; a++) {
      var t = 0; for (j = 0; j < d; j++) t += X[i * d + j] * axes[a][j];
      P[i * 3 + a] = t; maxAbs = Math.max(maxAbs, Math.abs(t));
    }
    for (i = 0; i < P.length; i++) P[i] = P[i] / maxAbs * 1.3;
    this.target = P;
    if (this.mesh) this.place(P, false);
    var ia = this.vin.indexOf(this.marked[0]), ib = this.vin.indexOf(this.marked[1]), dist = 0, all = 0, cnt = 0;
    for (j = 0; j < d; j++) dist += Math.pow(E[ia * d + j] - E[ib * d + j], 2);
    for (i = 0; i < n; i += 7) for (k = i + 3; k < n; k += 11) { var s2 = 0; for (j = 0; j < d; j++) s2 += Math.pow(E[i * d + j] - E[k * d + j], 2); all += Math.sqrt(s2); cnt++; }
    return { pair: Math.sqrt(dist), typical: all / cnt };
  };
  EmbView.prototype.place = function (P, jump) { this.pos = P; if (jump) this.cur = Float32Array.from(P); };
  EmbView.prototype.frame = function () {
    var m = new THREE.Matrix4(), n = this.vin.length, cur = this.cur, pos = this.pos;
    for (var i = 0; i < n * 3; i++) cur[i] += (pos[i] - cur[i]) * 0.15;
    for (i = 0; i < n; i++) {
      var big = this.markIdx.indexOf(i) >= 0 ? 2.4 : 1;
      m.makeScale(big, big, big); m.setPosition(cur[i * 3], cur[i * 3 + 1], cur[i * 3 + 2]); this.mesh.setMatrixAt(i, m);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    var lp = this.lineGeo.attributes.position.array, self = this;
    this.markIdx.forEach(function (k, j) { lp[j * 3] = cur[k * 3]; lp[j * 3 + 1] = cur[k * 3 + 1]; lp[j * 3 + 2] = cur[k * 3 + 2]; });
    this.lineGeo.attributes.position.needsUpdate = true;
    var w = this.el.clientWidth, h = this.el.clientHeight, v = new THREE.Vector3();
    this.markIdx.forEach(function (k, j) {
      v.set(cur[k * 3], cur[k * 3 + 1], cur[k * 3 + 2]).project(self.stage.camera);
      self.labels[j].style.transform = "translate(" + ((v.x + 1) / 2 * w + 10) + "px," + ((1 - v.y) / 2 * h - 10) + "px)";
    });
  };

  // ---------------------------------------------------------------- every parameter
  // One cube per parameter, block by block in the order the forward pass uses them. A cube's height is the size of its
  // weight and its colour the sign (blue negative, orange positive, grey near zero); a white glow is the size of the
  // parameter's latest change, so the glow shows where gradient descent is working right now. Long blocks (the token
  // tables) are folded into panels of 64 rows.
  var BLOCKS = { E: "token embeddings", P: "positions", Wq: "query", Wk: "key", Wv: "value", Wo: "attention output",
                 W1: "MLP in", b1: "", W2: "MLP out", b2: "", U: "unembedding", bU: "" };
  function ParamsView(el) { this.el = el; this.L = null; this.pending = null; var self = this; whenReady(function () { self.ready = true; if (self.L) self.build(); }); }
  ParamsView.prototype.setLayout = function (L) {
    if (this.L && this.L.d === L.d && this.L.size === L.size) return;
    this.L = L; this.prev = null; this.pending = null;
    if (this.ready) this.build();
  };
  ParamsView.prototype.build = function () {
    var self = this, L = this.L, FOLD = 64, GAP = 6, x = 0, cells = [];
    if (this.stage) { this.stage.scene.remove(this.mesh); this.mesh.geometry.dispose(); }
    else { this.stage = new Stage(this.el, 1); this.stage.hooks.push(function () { self.frame(); }); }
    this.labels && this.labels.forEach(function (d) { d.remove(); });
    this.labels = [];
    // each block is a grid of rows x cols, folded every 64 rows into panels side by side; blocks are packed onto shelves
    // (in forward-pass order) so that the whole network fills a roughly square area
    var blocks = L.names.map(function (name) {             // a bias (one row) is laid out as a column, so it folds too
      var t = L.off[name], o = t.rows === 1 ? { at: t.at, size: t.size, rows: t.cols, cols: 1, bias: true } : t, panels = Math.ceil(o.rows / FOLD);
      return { name: name, o: o, panels: panels, w: panels * (o.cols + 1.5) - 1.5, h: Math.min(o.rows, FOLD) };
    });
    var area = blocks.reduce(function (t, k) { return t + (k.w + GAP) * (k.h + GAP); }, 0), maxW = Math.sqrt(area) * 1.3;
    var shelfZ = 0, shelfH = 0, W = 0;
    blocks.forEach(function (k) {
      if (x > 0 && x + k.w > maxW) { shelfZ += shelfH + GAP * 1.6; x = 0; shelfH = 0; }
      k.x = x; k.z = shelfZ; x += k.w + GAP; shelfH = Math.max(shelfH, k.h); W = Math.max(W, x - GAP);
    });
    var D = shelfZ + shelfH;
    blocks.forEach(function (k) {
      var o = k.o;
      for (var i = 0; i < o.size; i++) {
        var r = Math.floor(i / o.cols), c = i % o.cols, pnl = Math.floor(r / FOLD);
        cells.push({ at: o.at + i, x: k.x + pnl * (o.cols + 1.5) + c - W / 2, z: k.z + (r % FOLD) - D / 2 });
      }
      if (BLOCKS[k.name]) self.labels.push({ text: BLOCKS[k.name] + " " + o.rows + "\u00d7" + o.cols, x: k.x + k.w / 2 - W / 2, z: k.z - 2 - D / 2 });
    });
    this.cells = cells; this.extent = Math.max(W, D, 40);
    this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(0.86, 1, 0.86), new THREE.MeshLambertMaterial({ color: 0xffffff }), cells.length);
    this.stage.scene.add(this.mesh);
    var s = this.extent; this.stage.camera.position.set(0, s * 0.72, s * 0.92); this.stage.controls.target.set(0, 0, 0);
    this.stage.camera.far = s * 6; this.stage.camera.updateProjectionMatrix();
    this.stage.controls.minDistance = s * 0.15; this.stage.controls.maxDistance = s * 4;   // never past the far plane
    this.tags = this.labels.map(function (l) { var d = document.createElement("span"); d.className = "v-label small"; d.textContent = l.text; self.el.appendChild(d); return d; });
    this.colors = this.colors || {};
    this.theme();
    if (this.pending) this.update(this.pending);
  };
  ParamsView.prototype.theme = function () {
    var dark = getComputedStyle(document.body).backgroundColor.match(/\d+/g).slice(0, 3).reduce(function (a, b) { return a + +b; }, 0) < 300;
    this.colors = { neg: new THREE.Color(dark ? "#3987e5" : "#2a78d6"), pos: new THREE.Color(dark ? "#d95926" : "#eb6834"),
                    mid: new THREE.Color(dark ? "#5b5a55" : "#c3c2b7"), glow: new THREE.Color(dark ? "#ffffff" : "#fffbe0") };
    if (this.last) this.update(this.last, true);
  };
  // params: the flat parameter array (TrainCore layout). Heights and colours from the weights; glow from the change
  // since the previous call.
  ParamsView.prototype.update = function (params, again) {
    if (!this.mesh) { this.pending = Float32Array.from(params); return; }
    var cells = this.cells, n = cells.length, m = new THREE.Matrix4(), c = new THREE.Color(), cs = this.colors, i;
    var prev = again ? this.prevSeen : this.prev, deltas = new Float32Array(n), big = 0, abs = new Float32Array(n), scale = 0;
    for (i = 0; i < n; i++) { var v = params[cells[i].at]; abs[i] = Math.abs(v); if (prev) deltas[i] = Math.abs(v - prev[cells[i].at]); }
    var sorted = Float32Array.from(abs).sort(); scale = sorted[Math.floor(n * 0.95)] || 1;   // 95th percentile weight = full height
    if (prev) { var ds = Float32Array.from(deltas).sort(); big = ds[Math.floor(n * 0.98)] || 0; }
    for (i = 0; i < n; i++) {
      var v2 = params[cells[i].at], t = Math.min(1, abs[i] / scale), h = 0.12 + 5 * t;
      m.makeScale(1, h, 1); m.setPosition(cells[i].x, h / 2, cells[i].z); this.mesh.setMatrixAt(i, m);
      c.copy(cs.mid).lerp(v2 < 0 ? cs.neg : cs.pos, Math.min(1, t * 1.4));
      if (big > 0) c.lerp(cs.glow, Math.min(1, deltas[i] / big) * 0.85);
      this.mesh.setColorAt(i, c);
    }
    this.mesh.instanceMatrix.needsUpdate = true; this.mesh.instanceColor.needsUpdate = true;
    if (!again) { this.prevSeen = this.prev; this.prev = Float32Array.from(params); this.last = this.prev; }
  };
  ParamsView.prototype.frame = function () {                 // keep the block labels over their blocks
    if (!this.tags) return;
    var w = this.el.clientWidth, h = this.el.clientHeight, v = new THREE.Vector3(), self = this;
    this.labels.forEach(function (l, j) {
      v.set(l.x, 0.5, l.z).project(self.stage.camera);
      var t = self.tags[j]; t.style.display = v.z < 1 ? "" : "none";
      t.style.transform = "translate(" + ((v.x + 1) / 2 * w) + "px," + ((1 - v.y) / 2 * h) + "px) translate(-50%, -100%)";
    });
  };

  root.TCTViews = {
    params: function (el) { return new ParamsView(el); },
    ready: ready, table: function (el, t) { return new TableView(el, t); }, emb: function (el, vin, marked) { return new EmbView(el, vin, marked); },
    retheme: function (views) { views.forEach(function (v) { if (v.theme) v.theme(); if (v.apply && v.mesh) v.apply(); }); }
  };
})(this);
