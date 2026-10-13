// Fractal curves: a Turing machine from curves.js, run by a trained network through the universal machine; every value
// it announces becomes a turn of the drawn line, checked against the machine run directly.
(function () {
  "use strict";
  var D = window.TCT_DATA, TB = window.TCT_TABLE, T = window.TCT, CV = window.TCT_CURVES;
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

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

  function direct(c, tape) {                               // the machine run directly: its announced values
    var m = {}; c.rules.forEach(function (r) { var a = r.split(" "); m[a[0] + " " + a[1]] = a; });
    var t = tape.split(""), h = 0, q = c.start, vals = [], steps = 0;
    while (q !== "qa" && q !== "qr") {
      var r = m[q + " " + (t[h] || "_")]; if (!r) break;
      t[h] = r[4]; h = Math.max(0, h + (r[5] === "L" ? -1 : r[5] === "R" ? 1 : 0)); q = r[3]; steps++;
      var v = c.emit(q); if (v !== null) vals.push(v);
    }
    return { vals: vals, steps: steps };
  }

  var curve = CV.curves[0], run, vals, want, playing = false, raf = null, elapsed = 0, matched = 0, frame = null;

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
    vals = []; matched = 0; elapsed = 0;
    frame = fit(curve.path(want.vals));                    // the frame the finished curve will fill (the drawing itself is only what the network computed)
    run = new T.FastRun(engine($("net").value), T.parseMachine(curve.start, curve.rules), tape, function (cfg) {
      var v = curve.emit(cfg.simState);
      if (v !== null) { if (v === want.vals[vals.length]) matched++; vals.push(v); }
    });
    $("encoded").textContent = run.encoded;
    $("play").textContent = "▶ Draw";
    render();
  }

  function tick() {
    if (!playing) return;
    var t0 = performance.now();
    do run.run(400000); while (!run.verdict && performance.now() - t0 < 28);
    elapsed += performance.now() - t0;
    render();
    if (run.verdict) { stop(); return; }
    raf = requestAnimationFrame(tick);
  }
  function stop() { playing = false; cancelAnimationFrame(raf); $("play").textContent = run && run.verdict ? "▶ Draw again" : "▶ Draw"; }

  function render() {
    $("turns").textContent = vals.length.toLocaleString() + " / " + want.vals.length.toLocaleString();
    $("turns-label").textContent = curve.id === "fibword" ? "Digits computed" : "Turns computed";
    $("match-label").textContent = (curve.id === "fibword" ? "Digits" : "Turns") + " matching the program run directly";
    $("passes").textContent = run.passes.toLocaleString();
    var left = vals.length > 20 ? Math.round(elapsed / vals.length * (want.vals.length - vals.length) / 1000) : null;
    $("rate").textContent = elapsed > 300 ? Math.round(run.passes / elapsed / 1000) + " million per second" + (!run.verdict && left !== null ? ", about " + left + " s to go" : "") : "";
    $("sim").textContent = run.simSteps.toLocaleString();
    $("utm").textContent = run.utmSteps.toLocaleString();
    $("distinct").textContent = run.distinct.toLocaleString();
    $("unseen").textContent = run.unseen; $("c-unseen").classList.toggle("warn", run.unseen > 0);
    $("disagree").textContent = run.disagree; $("c-dis").classList.toggle("warn", run.disagree > 0);
    $("match").textContent = matched.toLocaleString() + " / " + vals.length.toLocaleString();
    $("c-match").classList.toggle("warn", matched < vals.length);
    var v = $("verdict");
    if (run.verdict) {
      var ok = matched === want.vals.length && vals.length === want.vals.length && run.unseen === 0 && run.disagree === 0;
      v.textContent = ok ? "Done: all " + vals.length.toLocaleString() + (curve.id === "fibword" ? " digits" : " turns") + " computed by the " + ($("net").value === "8" ? "8,170" : "211,270") +
        "-parameter network, every one matching the machine run directly, from " + run.passes.toLocaleString() + " forward passes."
        : "This run differs from the machine run directly; see the checks below.";
      v.className = "verdict2 " + (ok ? "ok" : "bad");
    } else { v.textContent = ""; v.className = "verdict2"; }
    draw();
  }

  function fit(pts) {
    var xs = pts.map(function (p) { return p[0]; }), ys = pts.map(function (p) { return p[1]; });
    return { x0: Math.min.apply(null, xs), x1: Math.max.apply(null, xs), y0: Math.min.apply(null, ys), y1: Math.max.apply(null, ys) };
  }

  function mix(a, b, t) { return "rgb(" + a.map(function (x, i) { return Math.round(x + (b[i] - x) * t); }).join(",") + ")"; }
  function draw() {
    var cv = $("art"), dpr = window.devicePixelRatio || 1;
    if (frame) {                                           // the canvas takes the finished curve's proportions
      var aspect = Math.min(1.1, Math.max(0.42, (frame.y1 - frame.y0) / Math.max(1e-9, frame.x1 - frame.x0)));
      var tall = Math.min(Math.round(cv.clientWidth * aspect) + 48, Math.round(window.innerHeight * 0.75));
      if (cv.style.height !== tall + "px") cv.style.height = tall + "px";
    }
    var w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    var ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    if (!vals) return;
    var pts = curve.path(vals), f = frame, pad = 24;
    var s = Math.min((w - 2 * pad) / Math.max(1e-9, f.x1 - f.x0), (h - 2 * pad) / Math.max(1e-9, f.y1 - f.y0));
    var ox = (w - s * (f.x1 - f.x0)) / 2 - s * f.x0, oy = (h - s * (f.y1 - f.y0)) / 2 - s * f.y0;
    var dark = getComputedStyle(document.body).backgroundColor.match(/\d+/g).slice(0, 3).reduce(function (a, b) { return a + +b; }, 0) < 300;
    var c0 = dark ? [57, 135, 229] : [42, 120, 214], c1 = dark ? [217, 89, 38] : [235, 104, 52];
    var n = Math.max(1, want.vals.length + 1), chunk = Math.max(1, Math.ceil(pts.length / 160));
    ctx.lineWidth = Math.max(1.6, Math.min(3, s * 0.35)); ctx.lineJoin = ctx.lineCap = "round";
    for (var i = 0; i < pts.length - 1; i += chunk) {        // in short runs, each its own colour along the curve
      ctx.strokeStyle = mix(c0, c1, i / n); ctx.beginPath();
      ctx.moveTo(ox + s * pts[i][0], oy + s * pts[i][1]);
      for (var j = i + 1; j <= Math.min(i + chunk, pts.length - 1); j++) ctx.lineTo(ox + s * pts[j][0], oy + s * pts[j][1]);
      ctx.stroke();
    }
    if (!run.verdict && pts.length > 1) {                   // the pen
      var p = pts[pts.length - 1]; ctx.fillStyle = mix(c0, c1, (pts.length - 1) / n);
      ctx.beginPath(); ctx.arc(ox + s * p[0], oy + s * p[1], 5, 0, 7); ctx.fill();
    }
  }

  CV.curves.forEach(function (c) {
    var b = el("button", "chip", c.title); b.type = "button"; b.dataset.id = c.id;
    b.addEventListener("click", function () { choose(c.id); });
    $("curves").appendChild(b);
  });
  $("play").addEventListener("click", function () {
    if (playing) { stop(); return; }
    if (run.verdict) reset();
    playing = true; $("play").textContent = "❚❚ Pause"; tick();
  });
  $("reset").addEventListener("click", reset);
  $("bits").addEventListener("change", reset);
  $("net").addEventListener("change", reset);
  window.addEventListener("resize", draw);
  document.querySelector(".theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tct-theme", root.dataset.theme); } catch (e) {}
    draw();
  });
  var q = new URLSearchParams(location.search);            // curves.html?c=koch&k=8&net=64&play
  if (q.get("net") === "64") $("net").value = "64";
  choose(q.get("c") || "dragon");
  if (q.has("k") && Array.from($("bits").options).some(function (o) { return o.value === q.get("k"); })) { $("bits").value = q.get("k"); reset(); }
  if (q.has("play")) $("play").click();
})();
