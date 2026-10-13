// Fractal curves: a Turing machine from curves.js, run by a trained network through the universal machine; every value
// it announces becomes a turn of the drawn line, as it is computed. Nothing is drawn ahead of the network: the frame grows
// when the curve outgrows it, new segments are added to the canvas as they arrive, and the speed can be slowed down to
// one machine step, one universal-machine step, or one forward pass at a time. The machine is also run directly, only to
// check each value.
(function () {
  "use strict";
  var D = window.TCT_DATA, TB = window.TCT_TABLE, T = window.TCT, CV = window.TCT_CURVES;
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  var SEP = ":";

  // the two networks: the 8,170-parameter one (trained on the whole step table) and lag3
  var engines = {};
  function engine(which) {
    if (engines[which]) return engines[which];
    var data = D;
    if (which === "8") {
      var train = [];
      for (var k = 0; k < TB.labels.length; k++) train.push([0, 1, 2].map(function (t) { return TB.vin[TB.steps[3 * k + t]]; }).join("\u0001"));
      data = Object.assign({}, D, { weights: window.TCT_NET8.weights, heads: window.TCT_NET8.heads, train: train });
    }
    return (engines[which] = new T.Engine(data));
  }

  function direct(c, tape) {                               // the machine run directly: only to check the network's values
    var m = {}; c.rules.forEach(function (r) { var a = r.split(" "); m[a[0] + " " + a[1]] = a; });
    var t = tape.split(""), h = 0, q = c.start, vals = [];
    while (q !== "qa" && q !== "qr") {
      var r = m[q + " " + (t[h] || "_")]; if (!r) break;
      t[h] = r[4]; h = Math.max(0, h + (r[5] === "L" ? -1 : r[5] === "R" ? 1 : 0)); q = r[3];
      var v = c.emit(q); if (v !== null) vals.push(v);
    }
    return vals;
  }

  // speeds: how much to run between two screen updates, and how long to wait
  var SPEEDS = {
    fast: { label: "as fast as possible" },
    turns30: { label: "30 turns a second", turns: 1, wait: 33 },
    turns4: { label: "4 turns a second", turns: 1, wait: 250 },
    step: { label: "one machine step at a time", steps: 1, wait: 180 },
    utm: { label: "one universal-machine step at a time", utm: 1, wait: 220 },
    pass: { label: "one forward pass at a time", passes: 1, wait: 220 }
  };

  var curve = CV.curves[0], run, vals, want, total, playing = false, timer = null, elapsed = 0, matched = 0;
  var cfgNow = null, lastPass = null, view = null, drawn = 0, colors = null;

  function choose(id) {
    curve = CV.curves.filter(function (c) { return c.id === id; })[0] || CV.curves[0];
    document.querySelectorAll("#curves .chip").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.id === curve.id); });
    $("h-curve").textContent = curve.title; $("blurb").textContent = curve.blurb; $("how").textContent = curve.how;
    var rl = $("rules"); rl.innerHTML = ""; curve.rules.forEach(function (r) { rl.appendChild(el("div", "rule", r)); });
    var st = {}; curve.rules.forEach(function (r) { st[r.split(" ")[0]] = 1; }); $("nstates").textContent = Object.keys(st).length;
    var sizes = curve.sizes || [[8, "255 turns (seconds)"], [10, "1,023 turns (about half a minute)"], [12, "4,095 turns (two or three minutes)"]];
    var keep = $("bits").selectedIndex >= 0 ? $("bits").selectedIndex : 1;
    $("bits").innerHTML = ""; sizes.forEach(function (z) { $("bits").appendChild(new Option(z[1], z[0])); });
    $("bits").selectedIndex = Math.min(keep, sizes.length - 1);
    reset();
  }

  function reset() {
    stop();
    var k = +$("bits").value, tape = "_" + "0".repeat(k);
    want = direct(curve, tape);
    total = curve.expect(k).length;                        // how many values the program will announce, from its formula
    vals = []; matched = 0; elapsed = 0; cfgNow = null; lastPass = null; view = null; drawn = 0;
    run = new T.FastRun(engine($("net").value), T.parseMachine(curve.start, curve.rules), tape, function (cfg) {
      cfgNow = cfg;
      var v = curve.emit(cfg.simState);
      if (v !== null) { if (v === want[vals.length]) matched++; vals.push(v); }
    });
    $("encoded").textContent = run.encoded;
    $("play").textContent = "▶ Draw";
    clearArt(); render();
  }

  // ---------------------------------------------------------------- running, at the chosen speed
  function onePass() {                                     // one forward pass, keeping the window it read and what it wrote
    var k0 = run.k, n0 = run.n, win = run.tokens(k0, k0 + 3);
    run.run(1);
    lastPass = { window: win, out: run.n >= n0 ? run.tokens(n0, run.n) : [], k: run.passes };
  }
  function advance(sp) {
    if (sp.passes) { onePass(); return; }
    if (sp.utm) { var u = run.utmSteps, guard = 0; do onePass(); while (!run.verdict && run.utmSteps === u && ++guard < 100000); return; }
    run.stopEachStep = true;
    if (sp.steps) { var s = run.simSteps; while (!run.verdict && run.simSteps === s) run.run(1e7); }
    else { var t = vals.length; while (!run.verdict && vals.length === t) run.run(1e7); }
    run.stopEachStep = false; lastPass = null;
  }
  function tick() {
    if (!playing) return;
    var sp = SPEEDS[$("speed").value], t0 = performance.now();
    if (sp.wait) advance(sp);
    else { lastPass = null; do run.run(400000); while (!run.verdict && performance.now() - t0 < 28); }
    elapsed += performance.now() - t0;
    render();
    if (run.verdict) { stop(); return; }
    timer = sp.wait ? setTimeout(tick, sp.wait) : requestAnimationFrame(tick);
  }
  function stop() { playing = false; clearTimeout(timer); cancelAnimationFrame(timer); $("play").textContent = run && run.verdict ? "▶ Draw again" : "▶ Draw"; }

  // ---------------------------------------------------------------- the numbers
  function render() {
    var unit = curve.id === "fibword" ? "Digits" : "Turns";
    $("turns").textContent = vals.length.toLocaleString() + " / " + total.toLocaleString();
    $("turns-label").textContent = unit + " computed";
    $("match-label").textContent = unit + " matching the program run directly";
    $("passes").textContent = run.passes.toLocaleString();
    var fast = $("speed").value === "fast", left = vals.length > 20 ? Math.round(elapsed / vals.length * (total - vals.length) / 1000) : null;
    $("rate").textContent = fast && elapsed > 300 ? Math.round(run.passes / elapsed / 1000) + " million per second" + (!run.verdict && left !== null ? ", about " + left + " s to go" : "") : "";
    $("sim").textContent = run.simSteps.toLocaleString();
    $("utm").textContent = run.utmSteps.toLocaleString();
    $("distinct").textContent = run.distinct.toLocaleString();
    $("unseen").textContent = run.unseen; $("c-unseen").classList.toggle("warn", run.unseen > 0);
    $("disagree").textContent = run.disagree; $("c-dis").classList.toggle("warn", run.disagree > 0);
    $("match").textContent = matched.toLocaleString() + " / " + vals.length.toLocaleString();
    $("c-match").classList.toggle("warn", matched < vals.length);
    var v = $("verdict");
    if (run.verdict) {
      var ok = matched === want.length && vals.length === want.length && run.unseen === 0 && run.disagree === 0;
      v.textContent = ok ? "Done: all " + vals.length.toLocaleString() + " " + unit.toLowerCase() + " computed by the " + ($("net").value === "8" ? "8,170" : "211,270") +
        "-parameter network, every one matching the machine run directly, from " + run.passes.toLocaleString() + " forward passes."
        : "This run differs from the machine run directly; see the checks below.";
      v.className = "verdict2 " + (ok ? "ok" : "bad");
    } else { v.textContent = ""; v.className = "verdict2"; }
    inside(); draw();
  }

  // what the network is doing: the counting machine's tape and state, and, slowed down, the window it just read
  function inside() {
    var c = cfgNow, tape = $("in-tape");
    tape.innerHTML = "";
    var cells = c ? c.simTape : ("_" + "0".repeat(+$("bits").value)).split(""), head = c ? c.simHead : 0;
    for (var i = 0; i < cells.length; i++) {
      var s = cells[i], d = el("div", "pcell" + (s === "1" ? " one" : s === "0" ? " zero" : ""), s === "_" ? "" : s);
      if (i === head && !run.verdict) d.classList.add("head");
      tape.appendChild(d);
    }
    $("in-state").textContent = run.verdict ? "halted" : c ? c.simState : curve.start;
    $("in-n").textContent = c ? curve.count(c.simTape).toLocaleString() : "0";
    var said = c && curve.emit(c.simState) !== null;
    $("in-say").textContent = said ? "just announced: " + curve.label(curve.emit(c.simState)) : "";
    var box = $("in-pass");
    $("in-rings").hidden = !lastPass;
    if (!lastPass) { box.hidden = true; return; }
    box.hidden = false;
    var w = $("in-window"); w.innerHTML = "";
    lastPass.window.forEach(function (t, i) { var d = el("div", "wtok"); d.appendChild(el("span", "pos", ["left", "middle", "right"][i])); d.appendChild(el("b", null, show(t))); w.appendChild(d); });
    var o = $("in-out"); o.innerHTML = "";
    if (!lastPass.out.length) o.appendChild(el("span", "muted", "nothing: the universal machine halts"));
    lastPass.out.forEach(function (t) { var d = el("div", "wtok out"); d.appendChild(el("b", null, show(t))); o.appendChild(d); });
    // the transcript: the ring the window is reading, and the ring being written after it
    var B = run.BID, Tt = run.T, k = run.k, a = k, b;
    while (a > 0 && Tt[a] !== B) a--;
    for (b = a + 1; b < run.n && Tt[b] !== B; b++) {}
    var oldRing = run.tokens(a, b), newRing = run.tokens(b, run.n), headTok = null;
    ring($("in-old"), oldRing, k - a, k - a + 3);
    ring($("in-new"), newRing, newRing.length - lastPass.out.length, newRing.length);
    oldRing.forEach(function (t) { if (t.indexOf(SEP) > 0) headTok = t; });
    $("in-phase").textContent = headTok ? "The universal machine is in state " + headTok.split(SEP)[0] + ": " + T.phase(headTok.split(SEP)[0]) + "." : "";
  }
  function ring(box, toks, lo, hi) {                       // one ring of the transcript, cells lo..hi highlighted
    box.innerHTML = "";
    toks.forEach(function (t, i) {
      var head = t.indexOf(SEP) > 0, c = el("span", "rc" + (head ? " rh" : "") + (i >= lo && i < hi ? " rw" : ""), head ? t.split(SEP)[1] : t);
      if (head) c.title = "the universal machine's head, in state " + t.split(SEP)[0];
      box.appendChild(c);
    });
  }
  function show(t) { return t === "#" ? "#" : t; }

  // ---------------------------------------------------------------- the drawing: a pen that only moves forward
  function clearArt() {
    ["art", "pen"].forEach(function (id) { var cv = $(id), ctx = cv.getContext("2d"); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height); });
  }
  function mix(a, b, t) { return "rgb(" + a.map(function (x, i) { return Math.round(x + (b[i] - x) * t); }).join(",") + ")"; }
  function sizeCanvas(cv) {
    var dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; return true; }
    return false;
  }
  function fitView(pts, w, h) {                            // a frame around the curve so far, with room to grow
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    pts.forEach(function (p) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); });
    var cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, span = Math.max(x1 - x0, (y1 - y0) * w / h, 6) * 1.6, pad = 20;
    var s = (w - 2 * pad) / span;
    return { s: s, ox: w / 2 - s * cx, oy: h / 2 - s * cy };
  }
  function inView(p, v, w, h) { var X = v.ox + v.s * p[0], Y = v.oy + v.s * p[1]; return X >= 12 && X <= w - 12 && Y >= 12 && Y <= h - 12; }
  function draw(full) {
    var art = $("art"), pen = $("pen"), dpr = window.devicePixelRatio || 1;
    var resized = sizeCanvas(art); sizeCanvas(pen);
    var w = art.clientWidth, h = art.clientHeight, pts = curve.path(vals);
    var dark = getComputedStyle(document.body).backgroundColor.match(/\d+/g).slice(0, 3).reduce(function (a, b) { return a + +b; }, 0) < 300;
    colors = dark ? [[57, 135, 229], [217, 89, 38]] : [[42, 120, 214], [235, 104, 52]];
    var outside = view && pts.slice(drawn).some(function (p) { return !inView(p, view, w, h); });
    var ctx = art.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!view || outside || resized || full) {                // zoom out (or first frame): start the drawing over at the new scale
      view = fitView(pts, w, h); ctx.clearRect(0, 0, w, h); drawn = 0;
    }
    var v = view, n = Math.max(1, total + 1);
    ctx.lineWidth = Math.max(1.6, Math.min(3.5, v.s * 0.3)); ctx.lineJoin = ctx.lineCap = "round";
    for (var i = Math.max(0, drawn - 1); i < pts.length - 1; i++) {   // only the segments not yet on the canvas
      ctx.strokeStyle = mix(colors[0], colors[1], i / n); ctx.beginPath();
      ctx.moveTo(v.ox + v.s * pts[i][0], v.oy + v.s * pts[i][1]); ctx.lineTo(v.ox + v.s * pts[i + 1][0], v.oy + v.s * pts[i + 1][1]); ctx.stroke();
    }
    drawn = pts.length;
    var pc = pen.getContext("2d"); pc.setTransform(dpr, 0, 0, dpr, 0, 0); pc.clearRect(0, 0, w, h);
    if (!run.verdict && pts.length) {                       // the pen, on its own layer
      var p = pts[pts.length - 1], X = v.ox + v.s * p[0], Y = v.oy + v.s * p[1];
      pc.fillStyle = mix(colors[0], colors[1], (pts.length - 1) / n);
      pc.beginPath(); pc.arc(X, Y, 6, 0, 7); pc.fill();
      pc.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim(); pc.lineWidth = 2; pc.stroke();
    }
  }

  // ---------------------------------------------------------------- controls
  CV.curves.forEach(function (c) {
    var b = el("button", "chip", c.title); b.type = "button"; b.dataset.id = c.id;
    b.addEventListener("click", function () { choose(c.id); });
    $("curves").appendChild(b);
  });
  Object.keys(SPEEDS).forEach(function (k) { $("speed").appendChild(new Option(SPEEDS[k].label, k)); });
  $("play").addEventListener("click", function () {
    if (playing) { stop(); return; }
    if (run.verdict) reset();
    playing = true; $("play").textContent = "❚❚ Pause"; tick();
  });
  $("step").addEventListener("click", function () {
    if (run.verdict) reset();
    stop();
    var sp = SPEEDS[$("speed").value]; advance(sp.wait ? sp : SPEEDS.step); render();
  });
  $("reset").addEventListener("click", reset);
  $("bits").addEventListener("change", reset);
  $("net").addEventListener("change", reset);
  $("speed").addEventListener("change", function () { $("step").textContent = "One " + ({ pass: "forward pass", utm: "universal step", step: "machine step" }[$("speed").value] || "turn"); });
  window.addEventListener("resize", function () { draw(true); });
  document.querySelector(".theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tct-theme", root.dataset.theme); } catch (e) {}
    draw(true);
  });
  var q = new URLSearchParams(location.search);            // curves.html?c=koch&k=8&net=64&speed=step&play
  if (q.get("net") === "64") $("net").value = "64";
  if (SPEEDS[q.get("speed")]) $("speed").value = q.get("speed");
  $("speed").dispatchEvent(new Event("change"));
  choose(q.get("c") || "dragon");
  if (q.has("k") && Array.from($("bits").options).some(function (o) { return o.value === q.get("k"); })) { $("bits").value = q.get("k"); reset(); }
  if (q.has("play")) $("play").click();
})();
