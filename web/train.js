// Train it yourself: lag3's network, trained in the browser on the steps of the first N random machines, on the GPU
// (train-gpu.js) or across the processor's cores (train-core.js in Web Workers), then tested exactly on twelve
// programs it never saw.
(function () {
  "use strict";
  var T = window.TCT_TABLE, C = window.TrainCore, DATA = window.TCT_DATA;
  var WIDTHS = [64, 32, 16, 8, 6], L = C.layout(T.vin.length, T.vout.length, 8);   // the network's width, chosen on the page
  function layoutFor(d) { return C.layout(T.vin.length, T.vout.length, d); }
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  var STOPS = [10, 20, 50, 100, 200, 500, 1000, 2000, 2250, 2500, 2750, 3000, 3568, 4000, 16000];
  var NAMES = { increment: "increment", decrement: "decrement", invert: "invert", parity: "parity", div3: "divisible by 3",
                "unary-add": "unary addition", palindrome: "palindrome", rule90: "Rule 90", rule30: "Rule 30", rule110: "Rule 110",
                bb2: "busy beaver, 2 states", bb4: "busy beaver, 4 states" };
  var LR = 3e-3, NSTEPS = T.labels.length;

  // ---------------------------------------------------------------- the CPU trainer: train-core.js in Web Workers
  function workerMain() {
    var L, T, which, s;
    onmessage = function (e) {
      var m = e.data;
      if (m.type === "init") { T = m.table; L = TC.layout(m.nin, m.nout, m.d); s = TC.scratch(L); }
      else if (m.type === "set") which = m.which;
      else if (m.type === "grad") {
        var G = new Float32Array(L.size), r = TC.grad(L, m.p, G, T, which, m.n, s);
        postMessage({ G: G, loss: r.loss, correct: r.correct }, [G.buffer]);
      } else if (m.type === "predict") { var out = TC.predict(L, m.p, T, m.which); postMessage({ pred: out }, [out.buffer]); }
    };
  }
  function CpuTrainer(n) {
    this.L = L;
    var src = "var TC = (" + C.factory.toString() + ")();\n(" + workerMain.toString() + ")();";
    var url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
    this.workers = [];
    for (var i = 0; i < n; i++) {
      var w = new Worker(url);
      w.postMessage({ type: "init", table: { steps: T.steps, labels: T.labels }, nin: L.nin, nout: L.nout, d: L.d });
      this.workers.push(w);
    }
  }
  CpuTrainer.prototype.ask = function (w, msg) {
    return new Promise(function (res) { w.onmessage = function (e) { res(e.data); }; w.postMessage(msg); });
  };
  function chunks(arr, n) { var out = [], size = Math.ceil(arr.length / n); for (var i = 0; i < n; i++) out.push(arr.slice(i * size, (i + 1) * size)); return out; }
  CpuTrainer.prototype.setData = function (table, train) {
    this.nTrain = train.length;
    chunks(train, this.workers.length).forEach(function (c, i) { this.workers[i].postMessage({ type: "set", which: c }); }, this);
  };
  CpuTrainer.prototype.setParams = function (p) { this.p = Float32Array.from(p); this.M = new Float32Array(L.size); this.V = new Float32Array(L.size); this.t = 0; };
  CpuTrainer.prototype.step = async function (lr) {
    var self = this, rs = await Promise.all(this.workers.map(function (w) { return self.ask(w, { type: "grad", p: self.p, n: self.nTrain }); }));
    var G = rs[0].G, loss = 0, correct = 0;
    rs.forEach(function (r, i) { if (i) for (var j = 0; j < G.length; j++) G[j] += r.G[j]; loss += r.loss; correct += r.correct; });
    C.adam(this.p, G, this.M, this.V, ++this.t, lr);
    return { loss: loss / this.nTrain, correct: correct };
  };
  CpuTrainer.prototype.predictAll = async function () {
    var self = this, all = Array.from({ length: NSTEPS }, function (_, i) { return i; }), parts = chunks(all, this.workers.length);
    var rs = await Promise.all(this.workers.map(function (w, i) { return self.ask(w, { type: "predict", p: self.p, which: parts[i] }); }));
    var out = new Int32Array(NSTEPS), at = 0;
    rs.forEach(function (r) { out.set(r.pred, at); at += r.pred.length; });
    return out;
  };
  CpuTrainer.prototype.params = async function () { return Float32Array.from(this.p); };

  // ---------------------------------------------------------------- state
  var gpu = null, gpuDevice = null, cpu = null, eng = null, run = null, playing = false;
  var CORES = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));

  function machines() { return STOPS[+$("machines").value]; }
  function plan() {                                       // the training set for the chosen N, and what the tests need
    var N = machines(), train = [], inTrain = new Uint8Array(NSTEPS);
    for (var k = 0; k < NSTEPS; k++) if (T.first[k] >= 0 && T.first[k] < N) { train.push(k); inTrain[k] = 1; }
    var need = new Set(); Object.values(T.tests).forEach(function (v) { v.forEach(function (k) { need.add(k); }); });
    var missing = 0; need.forEach(function (k) { if (!inTrain[k]) missing++; });
    return { N: N, train: train, inTrain: inTrain, need: need.size, missing: missing };
  }

  function describe() {
    var p = plan();
    $("machines-out").textContent = p.N.toLocaleString();
    $("data-hint").textContent = p.train.length.toLocaleString() + " training steps, of the " + NSTEPS.toLocaleString() +
      " in the universal machine's table. The twelve programs need " + p.need.toLocaleString() + " steps; " +
      (p.missing ? p.missing.toLocaleString() + " of them are not in this training set." : "every one is in it.") +
      (p.N === 4000 ? " (This is lag3's training set.)" : p.N === 2500 ? " (In the experiment, the smallest that worked for every seed of the 64-wide network.)" :
       p.N === 16000 ? " (The whole table: no new step appeared after machine 7,000.)" : "");
    if ($("t-allin")) $("t-allin").textContent = p.train.length < NSTEPS ? "" :   // the step table says nothing until some steps are left out
      "At this setting every step is in training, so the whole stack is one colour. Slide “Train on the first” down to " +
      "about 1,000 machines and retrain: the steps it never saw sit on top, green where it gets them right anyway, red where it does not.";
  }

  function reset() {
    playing = false; $("train").textContent = "▶ Train";
    L = layoutFor(+$("width").value);
    $("hero-params").textContent = L.params.toLocaleString() + " parameters"; $("emb-width").textContent = L.d;
    $("p-count-cubes").textContent = L.params.toLocaleString();
    var vocab = L.off.E.size + L.off.U.size + L.off.bU.size;
    $("p-share").textContent = "the token tables are " + Math.round(100 * vocab / L.params) + "% of this network";
    var p = plan();
    run = { plan: p, seed: +$("seed").value || 0, max: +$("steps").value, step: 0, hist: [], evals: [], pred: null, ms: null, set: false, saved: null, fitAt: null };
    $("fit-note").textContent = "";
    $("t-step").textContent = "0"; ["t-loss", "t-fit", "t-unseen", "t-ms"].forEach(function (id) { $(id).textContent = "–"; });
    if ($("use")) { $("use").disabled = true; $("download").disabled = true; }
    describe(); programs(null); charts();
    if (tableView) { tableView.update(p.inTrain, null); }
    if (paramsView) { paramsView.setLayout(L); paramsView.update(C.init(L, run.seed)); }
  }

  async function engine() {                               // a trainer for the current width
    if ($("engine").value === "gpu") {
      if (!gpu || gpu.L.d !== L.d) gpu = new TrainGPU(gpuDevice, L);
      eng = gpu;
    } else {
      if (!cpu || cpu.L.d !== L.d) { if (cpu) cpu.workers.forEach(function (w) { w.terminate(); }); cpu = new CpuTrainer(CORES); }
      eng = cpu;
    }
    return eng;
  }

  var busy = false;                                       // the training loop is running (it finishes its step after a pause)
  async function train() {
    if (playing) { playing = false; $("train").textContent = "▶ Train"; return; }
    if (run.step >= run.max) reset();
    playing = true; busy = true; $("train").textContent = "❚❚ Pause";
    $("engine").disabled = true; $("machines").disabled = true; $("seed").disabled = true; $("width").disabled = true;
    try {
      if (!run.set) {
        await engine();
        eng.setData(T, run.plan.train);
        eng.setParams(C.init(L, run.seed));
        run.set = true; run.engine = $("engine").value;
        await evaluate();
      }
      var evalEvery = run.engine === "gpu" ? 10 : 10, last = 0;
      while (playing && run.step < run.max) {
        var t0 = performance.now(), r = await eng.step(LR), dt = performance.now() - t0;
        run.elapsed = (run.elapsed || 0) + dt;
        run.step++;
        run.ms = run.ms === null ? dt : run.ms * 0.9 + dt * 0.1;
        run.hist.push({ step: run.step, loss: r.loss, fit: r.correct / run.plan.train.length });
        if (paramsView && run.step % (L.d > 16 ? 8 : 2) === 0) paramsView.update(await eng.params());
        var fits = r.correct === run.plan.train.length;
        if (fits || run.step % evalEvery === 0 || run.step === run.max || run.step === 1) await evaluate();
        if (fits && $("stopfit").checked && trainingRight()) { run.fitAt = run.step; run.max = run.step; }
        if (performance.now() - last > 120 || run.step === run.max) { paint(); last = performance.now(); }
      }
    } catch (e) {
      $("engine-hint").textContent = "Training stopped: " + e.message;
      console.error(e);
    }
    playing = false; busy = false;
    $("train").textContent = run.step >= run.max ? "▶ Train again" : "▶ Train";
    $("engine").disabled = false; $("machines").disabled = false; $("seed").disabled = false; $("width").disabled = false;
    paint();
  }

  function trainingRight() {                              // after the update, is every training step still right?
    var p = run.plan; if (!run.pred) return false;
    for (var i = 0; i < p.train.length; i++) if (run.pred[p.train[i]] !== T.labels[p.train[i]]) return false;
    return true;
  }
  async function evaluate() {
    var pred = await eng.predictAll(), p = run.plan, lab = T.labels, uRight = 0, uAll = 0;
    for (var k = 0; k < NSTEPS; k++) if (!p.inTrain[k]) { uAll++; if (pred[k] === lab[k]) uRight++; }
    run.pred = pred;
    run.evals.push({ step: run.step, unseen: uAll ? uRight / uAll : null, uRight: uRight, uAll: uAll });
    programs(pred);
    if (tableView) tableView.update(p.inTrain, pred);
    if (embView) {
      var params = await eng.params(), E = params.subarray(L.off.E.at, L.off.E.at + L.off.E.size), d = embView.update(E, L.d);
      $("pair-dist").textContent = "distance between the two marked tokens: " + d.pair.toFixed(2) + " (a typical pair: " + d.typical.toFixed(2) + ")";
    }
    if (run.step > 0 && $("use")) { $("use").disabled = false; $("download").disabled = false; }
  }

  // ---------------------------------------------------------------- display
  function paint() {
    var h = run.hist[run.hist.length - 1], ev = run.evals[run.evals.length - 1];
    $("t-step").textContent = run.step.toLocaleString() + (run.fitAt ? "" : " / " + run.max.toLocaleString());
    if (run.fitAt) $("fit-note").textContent = "It got every one of its " + run.plan.train.length.toLocaleString() +
      " training steps right at Adam step " + run.fitAt.toLocaleString() + ", after " + (run.elapsed / 1000).toFixed(1) +
      " seconds of gradient descent, and stopped there.";
    if (h) { $("t-loss").textContent = h.loss < 0.001 ? h.loss.toExponential(2) : h.loss.toFixed(4);
             $("t-fit").textContent = (100 * h.fit).toFixed(2) + "%"; }
    if (ev && ev.uAll) $("t-unseen").textContent = ev.uRight.toLocaleString() + " / " + ev.uAll.toLocaleString();
    else if (ev) $("t-unseen").textContent = "none unseen";
    if (run.ms !== null) $("t-ms").textContent = run.ms < 1000 ? run.ms.toFixed(0) + " ms" : (run.ms / 1000).toFixed(1) + " s";
    charts();
  }

  function programs(pred) {
    var box = $("progs"), p = run.plan, ok = 0;
    box.innerHTML = "";
    Object.keys(T.tests).forEach(function (name) {
      var steps = T.tests[name], unseen = 0, wrong = 0, wrongUnseen = 0;
      steps.forEach(function (k) {
        if (!p.inTrain[k]) unseen++;
        if (pred && pred[k] !== T.labels[k]) { wrong++; if (!p.inTrain[k]) wrongUnseen++; }
      });
      var card = el("div", "prog" + (pred ? (wrong ? " no" : " yes") : ""));
      card.appendChild(el("b", null, NAMES[name] || name));
      card.appendChild(el("span", "sub2", steps.length.toLocaleString() + " steps, " + (unseen ? unseen + " never in training" : "all in training")));
      var v = el("span", "res", !pred ? "–" : wrong ? "✗ " + wrong + " step" + (wrong > 1 ? "s" : "") + " wrong" + (wrongUnseen ? " (" + wrongUnseen + " never seen)" : "") : "✓ runs correctly");
      card.appendChild(v);
      if (pred && !wrong) ok++;
      box.appendChild(card);
    });
    $("p-count").textContent = pred ? ok + " / 12" : "–";
  }

  function chart(cv, series, opts) {
    var w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
    if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    var ctx = cv.getContext("2d"), cs = getComputedStyle(document.documentElement);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    var padL = 44, padB = 20, padT = 8, padR = 10, X = function (s) { return padL + (w - padL - padR) * s / Math.max(1, run.max); };
    var Y = function (v) { return padT + (h - padT - padB) * (1 - (v - opts.lo) / (opts.hi - opts.lo)); };
    ctx.strokeStyle = cs.getPropertyValue("--rule").trim(); ctx.lineWidth = 1; ctx.fillStyle = cs.getPropertyValue("--muted").trim();
    ctx.font = "11px " + cs.getPropertyValue("--sans");
    opts.ticks.forEach(function (t) { var y = Y(t[0]); ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.fillText(t[1], 4, y + 4); });
    ctx.fillText("0", padL, h - 5); var e = run.max.toLocaleString(); ctx.fillText(e, w - padR - ctx.measureText(e).width, h - 5);
    series.forEach(function (s) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
      s.pts.forEach(function (p, i) { var x = X(p[0]), y = Y(Math.max(opts.lo, Math.min(opts.hi, p[1]))); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.stroke();
    });
  }
  function charts() {
    if (!run) return;
    var dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    var c1 = dark ? "#3987e5" : "#2a78d6", c2 = dark ? "#d95926" : "#eb6834";
    var lossPts = run.hist.map(function (x) { return [x.step, Math.log10(Math.max(1e-6, x.loss))]; });
    chart($("c-loss"), [{ color: c1, pts: lossPts }], { lo: -5, hi: 1, ticks: [[1, "10"], [0, "1"], [-1, "0.1"], [-2, "0.01"], [-3, "1e-3"], [-4, "1e-4"], [-5, "1e-5"]] });
    chart($("c-acc"), [{ color: c1, pts: run.hist.map(function (x) { return [x.step, x.fit * 100]; }) },
                       { color: c2, pts: run.evals.filter(function (x) { return x.unseen !== null; }).map(function (x) { return [x.step, x.unseen * 100]; }) }],
          { lo: 0, hi: 100, ticks: [[100, "100%"], [75, "75%"], [50, "50%"], [25, "25%"], [0, "0%"]] });
  }

  // ---------------------------------------------------------------- use, download
  function record(params) {
    var train = run.plan.train.map(function (k) { return [0, 1, 2].map(function (t) { return T.vin[T.steps[3 * k + t]]; }).join("\u0001"); });
    var correct = 0; if (run.pred) Object.keys(T.tests).forEach(function (n) { if (T.tests[n].every(function (k) { return run.pred[k] === T.labels[k]; })) correct++; });
    return { meta: { machines: run.plan.N, seed: run.seed, steps: run.step, programs: correct, date: new Date().toISOString(), engine: run.engine,
                     width: L.d, params: L.params },
             off: L.off, params: params, vinFull: T.vinFull, voutFull: T.voutFull, train: train };
  }
  // "Use your network" (section 5) is optional: a page may leave it out
  if ($("use")) {
  $("use").addEventListener("click", async function () {
    var rec = record(await eng.params());
    if (MyNet.save(rec) && MyNet.setActive(true)) {
      $("use-status").textContent = "Saved. The other pages now run your network (trained on " + rec.meta.machines.toLocaleString() +
        " machines, " + rec.meta.steps.toLocaleString() + " steps, " + rec.meta.programs + " of 12 programs correct), and say so at the top.";
    } else $("use-status").textContent = "This browser would not store the network (storage is blocked or full).";
  });
  $("download").addEventListener("click", async function () {
    var rec = record(await eng.params());
    var out = { meta: rec.meta, architecture: { d_model: L.d, heads: 2, mlp_hidden: L.hid }, vin: DATA.vin, vout: DATA.vout,
                weights: MyNet.weights(rec, DATA.vin.length, DATA.vout.length) };
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(out)], { type: "application/json" }));
    a.download = "tct-network-" + rec.meta.machines + "-machines-seed-" + rec.meta.seed + ".json"; a.click();
  });
  $("forget").addEventListener("click", function () {
    MyNet.setActive(false);
    $("use-status").textContent = "The other pages run the committed network again. (Your trained network stays saved until you train and save another.)";
  });
  function useStatus() {
    if (MyNet.active()) { var r = MyNet.load(); if (r) $("use-status").textContent = "The other pages currently run your network: " + r.meta.machines.toLocaleString() + " machines, seed " + r.meta.seed + ", " + r.meta.programs + " of 12 programs correct."; }
    else $("use-status").textContent = "The other pages currently run the committed network (lag3).";
  }
  }

  // ---------------------------------------------------------------- set up
  var tableView = null, embView = null, paramsView = null;
  $("machines").addEventListener("input", function () { if (!playing) reset(); });
  $("seed").addEventListener("change", function () { if (!playing) reset(); });
  $("steps").addEventListener("change", function () { if (!playing) { run.max = +$("steps").value; if (run.step >= run.max) reset(); else paint(); } });
  $("engine").addEventListener("change", function () { if (!playing) reset(); hintEngine(); });
  $("width").addEventListener("change", function () { if (!playing) reset(); });
  $("train").addEventListener("click", train);
  // each section's own Retrain button: stop any run, start a fresh one from the same settings, and stay where you are,
  // so the view beside the button shows training from step 0
  async function retrain() {
    if (playing) { playing = false; while (busy) await new Promise(function (r) { setTimeout(r, 30); }); }
    reset(); train();
  }
  document.querySelectorAll("[data-retrain]").forEach(function (b) { b.addEventListener("click", retrain); });
  $("reset").addEventListener("click", function () { if (!playing) reset(); });
  window.addEventListener("resize", charts);
  document.querySelector(".theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tct-theme", root.dataset.theme); } catch (e) {}
    charts(); TCTViews.retheme([tableView, embView, paramsView].filter(Boolean));
  });
  function hintEngine() {
    $("engine-hint").textContent = $("engine").value === "gpu"
      ? "On the GPU a step takes a few tens of milliseconds: lag3's 3,000 steps take a minute or two."
      : "On " + CORES + " processor cores a step takes a second or more; 300 steps is a good first run (the network usually fits its training set by step 200 to 700).";
  }

  (async function () {
    var sel = $("engine");
    WIDTHS.forEach(function (d) { $("width").appendChild(new Option(d + " wide: " + layoutFor(d).params.toLocaleString() + " parameters", d)); });
    $("width").value = "8";
    try { gpu = await TrainGPU.create(L); gpuDevice = gpu.device; sel.appendChild(new Option("this computer's GPU (WebGPU)", "gpu")); }
    catch (e) { gpu = null; }
    sel.appendChild(new Option("processor cores (" + CORES + ")", "cpu"));
    if (!gpu) { $("steps").value = "300"; }
    var q = new URLSearchParams(location.search);
    if (q.has("machines") && STOPS.indexOf(+q.get("machines")) >= 0) $("machines").value = STOPS.indexOf(+q.get("machines"));
    if (q.has("seed")) $("seed").value = +q.get("seed") || 0;
    if (["300", "1000", "3000"].indexOf(q.get("steps")) >= 0) $("steps").value = q.get("steps");
    if (q.get("engine") === "cpu") sel.value = "cpu";
    if (WIDTHS.indexOf(+q.get("width")) >= 0) $("width").value = q.get("width");
    hintEngine(); reset(); if ($("use")) useStatus();
    var sel2 = $("t-filter");
    Object.keys(T.tests).forEach(function (n) { sel2.appendChild(new Option("only the steps " + (NAMES[n] || n) + " needs", n)); });
    try {
      tableView = TCTViews.table($("v-table"), T); embView = TCTViews.emb($("v-emb"), T.vin, ["go_aS:I", "go_bS:I"]);
      paramsView = TCTViews.params($("v-params")); paramsView.setLayout(L); paramsView.update(C.init(L, run.seed));
      tableView.update(run.plan.inTrain, null);
      sel2.addEventListener("change", function () { tableView.setFilter(sel2.value ? T.tests[sel2.value] : null); });
      TCTViews.ready.catch(function () { $("v-table").innerHTML = $("v-emb").innerHTML = '<p class="v-msg">The 3D views need three.js from the network, which did not load.</p>'; });
    } catch (e) { console.error(e); }
    if (q.has("play")) train();
  })();
})();
