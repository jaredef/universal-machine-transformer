// It draws: lag3's trained network runs the universal machine, which runs a cellular-automaton machine.
(function () {
  "use strict";
  var D = window.TCT_DATA, T = window.TCT, engine = new T.Engine(D);
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  function caRules(rule) {
    var f = function (a, b, c) { return (rule >> (a * 4 + b * 2 + c)) & 1; }, R = ["g _ -> c00 _ R"];
    [0, 1].forEach(function (a) { [0, 1].forEach(function (b) {
      [0, 1].forEach(function (x) { R.push("c" + a + b + " " + x + " -> c" + b + x + " " + f(a, b, x) + " R"); });
      R.push("c" + a + b + " _ -> e" + b + " " + f(a, b, 0) + " R");
    }); });
    [0, 1].forEach(function (b) { R.push("e" + b + " _ -> back " + f(b, 0, 0) + " L"); });
    return R.concat(["back 0 -> back 0 L", "back 1 -> back 1 L", "back _ -> g _ S"]);
  }
  function reference(rule, G) {
    var f = function (a, b, c) { return (rule >> (a * 4 + b * 2 + c)) & 1; }, row = "1", rows = [row];
    while (rows.length < G) {
      var p = "00" + row + "00", r = "";
      for (var i = 1; i < p.length - 1; i++) r += f(+p[i - 1], +p[i], +p[i + 1]);
      rows.push(row = r);
    }
    return rows;
  }

  var rule = 90, G = 32, run = null, rows = [], want = [], live = null, playing = false, budget = 400000, elapsed = 0;

  function reset() {
    playing = false; $("draw").textContent = "▶ Draw";
    G = +$("rows").value;
    var machine = T.parseMachine("g", caRules(rule));
    rows = ["1"]; want = reference(rule, G); live = null; elapsed = 0;
    run = new T.FastRun(engine, machine, "_1", function (cfg) {
      live = cfg;
      if (cfg.simState === "g") rows.push(cfg.simTape.slice(1).join("").replace(/_+$/, ""));
    });
    document.querySelectorAll("#rules .chip").forEach(function (b) { b.setAttribute("aria-pressed", +b.dataset.rule === rule); });
    $("rule-input").value = rule;
    $("encoded").textContent = run.encoded;
    var mr = $("machine-rules"); mr.innerHTML = "";
    caRules(rule).forEach(function (r) { mr.appendChild(el("div", "rule", r)); });
    $("status").textContent = "Press Draw.";
    render();
  }

  function frame() {
    if (!playing) return;
    var start = performance.now();
    run.run(budget);
    var dt = performance.now() - start;
    elapsed += dt;
    budget = Math.max(50000, Math.min(4000000, Math.round(budget * (24 / Math.max(dt, 1)))));   // aim for ~24 ms of work
    if (rows.length >= G) { playing = false; $("draw").textContent = "▶ Draw again"; }
    render();
    if (playing) requestAnimationFrame(frame);
  }

  function render() {
    var n = Math.min(rows.length, G);
    $("rows-done").textContent = n + " / " + G;
    $("passes").textContent = run.passes.toLocaleString();
    $("rate").textContent = elapsed > 200 ? Math.round(run.passes / elapsed / 1000) + " million per second" : "";
    $("utm").textContent = run.utmSteps.toLocaleString();
    $("sim").textContent = run.simSteps.toLocaleString();
    $("distinct").textContent = run.distinct.toLocaleString();
    $("unseen").textContent = run.unseen; $("c-unseen").classList.toggle("warn", run.unseen > 0);
    $("disagree").textContent = run.disagree; $("c-dis").classList.toggle("warn", run.disagree > 0);
    var correct = 0;
    for (var i = 0; i < n; i++) if (rows[i] === want[i]) correct++;
    if (n >= G) $("status").textContent = correct === G
      ? "Done: all " + G + " rows match the automaton computed directly, from " + run.passes.toLocaleString() + " forward passes."
      : (G - correct) + " of " + G + " rows differ from the automaton computed directly.";
    else if (playing) $("status").textContent = "Drawing row " + (n + 1) + "…";
    paint(n);
  }

  function paint(n) {
    var cv = $("canvas"), wrap = cv.parentElement;
    var cols = 2 * G - 1, cssW = Math.min(wrap.clientWidth - 36, 820), cell = Math.max(2, Math.floor(cssW / cols));
    var w = cell * cols, h = cell * G, dpr = window.devicePixelRatio || 1;
    if (cv.width !== w * dpr || cv.height !== h * dpr) {
      cv.width = w * dpr; cv.height = h * dpr; cv.style.width = w + "px"; cv.style.height = h + "px";
    }
    var ctx = cv.getContext("2d"), cs = getComputedStyle(document.documentElement);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = cs.getPropertyValue("--soft").trim(); ctx.fillRect(0, 0, w, h);
    var on = cs.getPropertyValue("--ink").trim(), bad = cs.getPropertyValue("--bad").trim();
    for (var g = 0; g < n; g++) {
      var row = rows[g], off = (G - 1 - g) * cell, ok = row === want[g];
      ctx.fillStyle = ok ? on : bad;
      for (var j = 0; j < row.length; j++) if (row[j] === "1") ctx.fillRect(off + j * cell, g * cell, cell - (cell > 4 ? 1 : 0), cell - (cell > 4 ? 1 : 0));
    }
    if (n < G && live && playing) {                          // where the described machine's head is, on the row in progress
      ctx.fillStyle = cs.getPropertyValue("--accent-2").trim();
      var x = (G - 1 - n) * cell + Math.max(0, live.simHead - 1) * cell;
      ctx.fillRect(x, n * cell, cell, cell);
    }
    var ticks = $("ticks"); ticks.style.height = h + "px"; ticks.innerHTML = "";
    for (var r = 0; r < n; r++) {
      var t = el("span", rows[r] === want[r] ? "ok" : "no", rows[r] === want[r] ? "✓" : "✗");
      t.style.top = r * cell + "px"; t.style.height = cell + "px"; t.style.fontSize = Math.min(13, Math.max(7, cell)) + "px";
      ticks.appendChild(t);
    }
  }

  $("rules").addEventListener("click", function (e) {
    var b = e.target.closest(".chip"); if (!b) return; rule = +b.dataset.rule; reset();
  });
  $("rule-input").addEventListener("change", function () {
    var v = Math.round(+$("rule-input").value);
    if (v >= 0 && v <= 255) { rule = v; reset(); }
  });
  $("rows").addEventListener("change", reset);
  $("draw").addEventListener("click", function () {
    if (playing) { playing = false; $("draw").textContent = "▶ Draw"; render(); return; }
    if (rows.length >= G) reset();
    playing = true; $("draw").textContent = "❚❚ Pause"; requestAnimationFrame(frame);
  });
  $("reset").addEventListener("click", reset);
  window.addEventListener("resize", function () { paint(Math.min(rows.length, G)); });
  document.querySelector(".theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tct-theme", root.dataset.theme); } catch (e) {}
    paint(Math.min(rows.length, G));
  });
  var q = new URLSearchParams(location.search);               // draw.html?rule=30&rows=48&play
  if (q.has("rule") && +q.get("rule") >= 0 && +q.get("rule") <= 255) rule = Math.round(+q.get("rule"));
  if (["16", "32", "48"].indexOf(q.get("rows")) >= 0) $("rows").value = q.get("rows");
  reset();
  if (q.has("play")) $("draw").click();
})();
