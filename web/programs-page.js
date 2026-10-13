// Programs it never saw: run a program from programs.js on the trained network, through the universal machine, and
// check every configuration against the program run directly.
(function () {
  "use strict";
  var D = window.TCT_DATA, T = window.TCT, PR = window.TCT_PROGRAMS, engine = new T.Engine(D);
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  function direct(p, tape) {                               // the program run directly (one-way tape)
    var m = {}; p.rules.forEach(function (r) { var a = r.split(" "); m[a[0] + " " + a[1]] = a; });
    var t = tape.split(""), h = 0, q = p.start, out = [{ state: q, head: h, tape: t.join("").replace(/_+$/, "") }];
    while (q !== "qa" && q !== "qr" && out.length < 2e6) {
      var r = m[q + " " + (t[h] || "_")]; if (!r) { q = "qr"; out.push({ state: q, head: h, tape: t.join("").replace(/_+$/, "") }); break; }
      t[h] = r[4]; h = Math.max(0, h + (r[5] === "L" ? -1 : r[5] === "R" ? 1 : 0)); q = r[3];
      out.push({ state: q, head: h, tape: t.join("").replace(/_+$/, "") });
    }
    return out;
  }

  var prog = PR.programs[0], param, want, run, got, values, playing = false, timer = null, elapsed = 0, matched = 0;

  function parseParam() {
    var raw = $("param").value.trim(), pp = prog.param;
    if (typeof pp.value === "number") {
      var n = Math.round(+raw);
      if (!(n >= pp.min && n <= pp.max)) throw new Error("a whole number from " + pp.min + " to " + pp.max);
      return n;
    }
    if (!new RegExp(pp.pattern).test(raw)) throw new Error(prog.id === "add" ? "like 1000+99 (a up to 5 digits, b up to 3)" : "1 to 24 bits of 0 and 1");
    return raw;
  }

  function choose(id) {
    prog = PR.programs.filter(function (p) { return p.id === id; })[0] || PR.programs[0];
    $("param").value = prog.param.value; $("param-label").textContent = prog.param.label;
    document.querySelectorAll("#progs .chip").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.id === prog.id); });
    $("h-prog").textContent = prog.title; $("blurb").textContent = prog.blurb; $("how").textContent = prog.how;
    var rl = $("rules"); rl.innerHTML = ""; prog.rules.forEach(function (r) { rl.appendChild(el("div", "rule", r)); });
    var states = {}; prog.rules.forEach(function (r) { states[r.split(" ")[0]] = 1; }); $("nstates").textContent = Object.keys(states).length;
    reset();
  }

  function reset() {
    stop();
    try { param = parseParam(); $("param-hint").textContent = ""; }
    catch (e) { $("param-hint").textContent = "Input: " + e.message; return; }
    var tape = prog.input(param);
    want = direct(prog, tape);
    got = [{ state: prog.start, head: 0, tape: tape.replace(/_+$/, "") }];
    values = []; matched = 1; elapsed = 0;
    run = new T.FastRun(engine, T.parseMachine(prog.start, prog.rules), tape, function (cfg) {
      got.push({ state: cfg.simState, head: cfg.simHead, tape: cfg.simTape.join("").replace(/_+$/, "") });
    });
    run.stopEachStep = true;
    run.onHalt = function (cfg) { got.push({ state: run.verdict === "ACCEPT" ? "qa" : "qr", head: null, tape: cfg.simTape.join("").replace(/_+$/, ""), final: true }); };
    $("encoded").textContent = run.encoded;
    $("play").textContent = "▶ Run";
    render();
  }

  function advance() {                                     // one machine step; true when a new value appeared
    if (run.verdict) return false;
    var before = got.length;
    while (!run.verdict && got.length === before) run.run(20000000);
    var fresh = false;
    for (var i = before; i < got.length; i++) {
      var g = got[i], w = want[i];
      if (w && g.state === w.state && g.tape === w.tape && (g.final || g.head === w.head)) matched++;
      if (!g.final) { var v = prog.value(g); if (v !== null) { values.push(v); fresh = true; } }
    }
    return fresh;
  }

  function tick() {
    if (!playing) return;
    var t0 = performance.now(), watch = $("speed").value === "watch";
    while (!run.verdict && performance.now() - t0 < 30) { if (advance() && watch) break; }
    elapsed += performance.now() - t0;
    render();
    if (run.verdict) { stop(); return; }
    timer = watch ? setTimeout(tick, 90) : requestAnimationFrame(tick);
  }
  function stop() { playing = false; clearTimeout(timer); cancelAnimationFrame(timer); $("play").textContent = run && run.verdict ? "▶ Run again" : "▶ Run"; }

  function render() {
    if (!run) return;
    var last = values[values.length - 1];
    $("bigval").textContent = last === undefined ? "–" : prog.show(last);
    $("passes").textContent = run.passes.toLocaleString();
    $("rate").textContent = elapsed > 300 ? Math.round(run.passes / elapsed / 1000) + " million per second" : "";
    $("sim").textContent = (got.length - 1).toLocaleString();
    $("utm").textContent = run.utmSteps.toLocaleString();
    $("distinct").textContent = run.distinct.toLocaleString();
    $("unseen").textContent = run.unseen; $("c-unseen").classList.toggle("warn", run.unseen > 0);
    $("disagree").textContent = run.disagree; $("c-dis").classList.toggle("warn", run.disagree > 0);
    $("match").textContent = matched.toLocaleString() + " / " + got.length.toLocaleString();
    $("c-match").classList.toggle("warn", matched < got.length);
    var cur = got[got.length - 1], tape = $("tape"), t = cur.tape || "_";
    tape.innerHTML = "";
    var first = t.search(/[01]/), from = Math.max(0, Math.min(first < 0 ? 0 : first, cur.final ? Infinity : cur.head) - 2);
    if (from > 0) tape.appendChild(el("div", "pskip", from + " blank cells"));
    for (var i = from; i < Math.max(t.length, (cur.head || 0) + 1); i++) {
      var s = t[i] || "_", c = el("div", "pcell" + (s === "1" ? " one" : s === "0" ? " zero" : ""), s === "_" ? "" : s);
      if (i === cur.head && !cur.final) { c.classList.add("head"); c.title = "state " + cur.state; }
      tape.appendChild(c);
    }
    var v = $("verdict");
    if (run.verdict) {
      var ok = matched === got.length && got.length === want.length && run.unseen === 0 && run.disagree === 0;
      v.textContent = ok ? "Halted: " + prog.result(cur.tape) + ". Every one of its " + (got.length - 1).toLocaleString() +
        " steps matches the program run directly, from " + run.passes.toLocaleString() + " forward passes of the network."
        : "This run differs from the program run directly; see the checks below.";
      v.className = "verdict2 " + (ok ? "ok" : "bad");
    } else { v.textContent = ""; v.className = "verdict2"; }
    plot();
  }

  function plot() {
    var cv = $("plot"), w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
    if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    var ctx = cv.getContext("2d"), cs = getComputedStyle(document.documentElement);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    var ink = cs.getPropertyValue("--ink").trim(), acc = cs.getPropertyValue("--accent").trim(), rule = cs.getPropertyValue("--rule").trim(), muted = cs.getPropertyValue("--muted").trim();
    var pts = values.map(function (v) { return prog.plot(v); });
    if (!pts.length) return;
    if (typeof pts[0] === "string") {                      // rows of bits, one per pass, newest at the bottom
      var cols = Math.max.apply(null, pts.map(function (s) { return s.length; })), rows = pts.length, total = want.filter(function (c) { return prog.value(c) !== null; }).length || rows;
      var cell = Math.max(2, Math.min(14, Math.floor((w - 4) / cols), Math.floor(520 / Math.max(total, 1))));
      if (cv.style.height !== total * cell + "px") { cv.style.height = total * cell + "px"; return plot(); }
      var x0 = (w - cell * cols) / 2;
      ctx.fillStyle = ink;
      pts.forEach(function (s, r) { for (var j = 0; j < s.length; j++) if (s[j] === "1") ctx.fillRect(x0 + j * cell, r * cell, Math.max(1, cell - (cell > 4 ? 1 : 0)), Math.max(1, cell - (cell > 4 ? 1 : 0))); });
      return;
    }
    if (cv.style.height !== "220px") { cv.style.height = "220px"; return plot(); }
    var all = want.map(function (c) { return prog.value(c); }).filter(function (v) { return v !== null; }).map(prog.plot);
    var n = all.length, hi = Math.max.apply(null, all), lo = Math.min(0, Math.min.apply(null, all));
    var X = function (i) { return 40 + (w - 50) * (n > 1 ? i / (n - 1) : 0); }, Y = function (v) { return 8 + (h - 24) * (1 - (v - lo) / ((hi - lo) || 1)); };
    ctx.strokeStyle = rule; ctx.lineWidth = 1; ctx.fillStyle = muted; ctx.font = "11px " + cs.getPropertyValue("--sans");
    [lo, (lo + hi) / 2, hi].forEach(function (v) { var y = Y(v); ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(w - 10, y); ctx.stroke(); ctx.fillText(Math.round(v).toLocaleString(), 0, y + 4); });
    ctx.strokeStyle = acc; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach(function (v, i) { if (i) ctx.lineTo(X(i), Y(v)); else ctx.moveTo(X(i), Y(v)); });
    ctx.stroke();
    var k = pts.length - 1; ctx.fillStyle = acc; ctx.beginPath(); ctx.arc(X(k), Y(pts[k]), 4, 0, 7); ctx.fill();
  }

  PR.programs.forEach(function (p) {
    var b = el("button", "chip", p.title); b.type = "button"; b.dataset.id = p.id;
    b.addEventListener("click", function () { choose(p.id); });
    $("progs").appendChild(b);
  });
  $("play").addEventListener("click", function () {
    if (playing) { stop(); return; }
    if (!run || run.verdict) reset();
    if (!run) return;
    playing = true; $("play").textContent = "❚❚ Pause"; tick();
  });
  $("step").addEventListener("click", function () { if (!run || run.verdict) reset(); stop(); advance(); render(); });
  $("reset").addEventListener("click", reset);
  $("param").addEventListener("change", reset);
  window.addEventListener("resize", plot);
  document.querySelector(".theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tct-theme", root.dataset.theme); } catch (e) {}
    plot();
  });
  var q = new URLSearchParams(location.search);            // programs.html?p=collatz&n=871&play
  choose(q.get("p") || "collatz");
  if (q.has("n")) { $("param").value = q.get("n"); reset(); }
  if (q.has("play")) $("play").click();
})();
