// The page: choose or write a machine, give it an input, and watch lag3's trained network run it.
(function () {
  "use strict";
  var D = window.TCT_DATA, T = window.TCT;
  var engine = new T.Engine(D);
  var SIM = ["0", "1", "_"];
  var MAX_PASSES = 30000000;
  var OWN_DEFAULT = "# add two to a binary number (give it a leading 0)\n" +
    "q0 0 -> q0 0 R\nq0 1 -> q0 1 R\nq0 _ -> q1 _ L\n" +
    "q1 0 -> q2 0 L\nq1 1 -> q2 1 L\n" +
    "q2 0 -> qa 1 S\nq2 1 -> q3 0 L\nq3 1 -> q3 0 L\nq3 0 -> qa 1 S";

  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function fmtN(n) { return n.toLocaleString(); }

  var current = { name: "increment", lines: null, start: null }, machine = null, run = null;
  var playing = false, lastTick = 0;

  // ---------------------------------------------------------------- choosing a machine

  function buildChips() {
    var box = $("machines");
    Object.keys(D.library).forEach(function (name) {
      var b = el("button", "chip", name); b.type = "button"; b.dataset.name = name; b.title = D.library[name].description;
      box.appendChild(b);
    });
    var own = el("button", "chip own", "write your own"); own.type = "button"; own.dataset.name = "own"; box.appendChild(own);
    box.addEventListener("click", function (e) { var b = e.target.closest(".chip"); if (b) choose(b.dataset.name); });
  }

  function choose(name) {
    stop();
    current.name = name;
    document.querySelectorAll(".chip").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.name === name); });
    var own = name === "own";
    $("editor").hidden = !own;
    if (own) {
      if (!$("rules").value) $("rules").value = OWN_DEFAULT;
      $("input").value = $("input").value || "0101";
      $("input-hint").textContent = "Your machine runs on a tape that starts at the first cell; a left move there stays put.";
    } else {
      $("input").value = D.examples[name];
      $("input-hint").textContent = D.library[name].description + ".";
    }
    reset();
  }

  function currentMachine() {
    if (current.name !== "own") {
      var L = D.library[current.name];
      return { m: T.parseMachine(L.start, L.rules), lines: L.rules };
    }
    var lines = $("rules").value.split("\n");
    var first = lines.map(function (l) { return l.split("#")[0].trim(); }).filter(Boolean)[0];
    if (!first) throw new Error("write at least one rule");
    return { m: T.parseMachine(first.split(/\s+/)[0], lines), lines: lines };
  }

  // ---------------------------------------------------------------- running

  function reset() {
    stop();
    $("rule-error").textContent = "";
    var input = $("input").value.trim();
    try {
      if (!/^[01_]{1,48}$/.test(input)) throw new Error("the input must be 1 to 48 symbols, each 0, 1, or _");
      var cm = currentMachine();
      machine = cm.m; current.lines = cm.lines.filter(function (l) { return l.split("#")[0].trim(); });
      run = new T.Run(engine, machine, input);
    } catch (err) {
      $("rule-error").textContent = err.message;
      if (current.name !== "own") $("input-hint").textContent = err.message;
      run = null;
    }
    render();
  }

  function advance(mode) {
    if (!run || run.verdict) return;
    if (mode === "fast") { run.advance("end", 400000); }
    else run.advance(mode, 5000000);
    if (!run.verdict && run.passes >= MAX_PASSES) { run.verdict = "TIMEOUT"; run.result = ""; }
  }

  function tick(t) {
    if (!playing) return;
    var mode = $("speed").value, wait = mode === "pass" ? 160 : mode === "sim" ? 420 : 0;
    if (t - lastTick >= wait) { lastTick = t; advance(mode); render(); }
    if (run && run.verdict) { stop(); render(); return; }
    requestAnimationFrame(tick);
  }
  function play() {
    if (!run) return;
    if (run.verdict) reset();
    playing = true; $("play").textContent = "❚❚ Pause"; lastTick = 0; requestAnimationFrame(tick);
  }
  function stop() { playing = false; $("play").textContent = "▶ Run"; }

  // ---------------------------------------------------------------- drawing

  function render() {
    var ok = !!run;
    ["step-pass", "step-utm", "step-sim", "to-end"].forEach(function (id) { $(id).disabled = !ok || !!run.verdict; });
    if (!ok) return;
    var cfg = run.config;
    renderMachine(cfg);
    renderUniversal(cfg);
    renderNetwork();
  }

  function renderMachine(cfg) {
    $("sim-steps").textContent = fmtN(run.simSteps);
    var state = run.verdict ? (run.verdict === "ACCEPT" ? "qa" : run.verdict === "REJECT" ? "qr" : cfg.simState) : cfg.simState;
    $("sim-state").textContent = state;
    var tape = $("sim-tape"); tape.innerHTML = "";
    var n = Math.max(cfg.simTape.length, cfg.simHead + 1);
    for (var i = 0; i < n; i++) {
      var s = cfg.simTape[i] || "_";
      tape.appendChild(el("span", "cell" + (s === "_" ? " blank" : "") + (i === cfg.simHead && !run.verdict ? " head" : ""), s));
    }
    // the rules, with the one in use highlighted
    var table = $("rule-table"); table.innerHTML = "";
    var liveKey = null;
    if (cfg.selected && cfg.selected.entry >= 0) liveKey = machine.order[cfg.selected.entry] + " " + SIM[cfg.selected.block];
    current.lines.forEach(function (line) {
      var p = line.split("#")[0].trim().split(/\s+/), k = p[0] + " " + p[1];
      table.appendChild(el("div", "rule" + (k === liveKey ? " live" : p[0] === state ? " cur" : ""), p.join(" ")));
    });
    var v = $("verdict"); v.innerHTML = "";
    if (run.verdict) {
      var box = el("div", "box");
      if (run.verdict === "TIMEOUT") {
        box.appendChild(el("span", "no", "Stopped"));
        box.appendChild(el("span", null, "no halt after " + fmtN(run.passes) + " forward passes; your machine may run forever"));
      } else {
        box.appendChild(el("b", null, run.verdict === "ACCEPT" ? "Accepted" : "Rejected"));
        box.appendChild(el("span", null, "tape " + run.result));
        box.appendChild(el("span", run.correct ? "ok" : "no", run.correct
          ? "✓ the same as running your machine directly"
          : "✗ running your machine directly gives " + run.reference.verdict.toLowerCase() + ", tape " + run.reference.tape));
      }
      v.appendChild(box);
    }
  }

  function renderUniversal(cfg) {
    $("utm-steps").textContent = fmtN(run.utmSteps);
    $("utm-phase").textContent = run.verdict === "ACCEPT" ? "finished: your machine accepted"
      : run.verdict === "REJECT" ? "finished: your machine rejected" : T.phase(cfg.state);
    var box = $("utm-tape"); box.innerHTML = "";
    var entry = -1, inSel = false, block = 0;
    cfg.cells.forEach(function (c, i) {
      var cls = "u";
      if (cfg.dollar >= 0 && i > cfg.dollar) cls += " sim" + (c === "_" ? " blank" : "");
      else if (i === cfg.dollar) cls += " dollar";
      else if (c === "_") cls += " blank";            // the cell the tape grew by, left of the program
      else {
        if (c === "*" || c === "|") { entry++; block = 0; }
        cls += entry % 2 === 0 ? " ea" : " eb";
        if (c === "*") cls += " star";
        if (cfg.selected && entry === cfg.selected.entry && block === cfg.selected.block && c !== "*" && c !== "|") cls += " sel";
        if (c === ";") block++;
      }
      if (i === cfg.head) cls += " head";
      var u = el("span", cls, c);
      if (i === cfg.head) { var tag = el("span", "tag", cfg.state); u.appendChild(tag); }
      box.appendChild(u);
    });
  }

  function show(name) {
    var toks = D.emits[name];
    return toks.length ? toks.join("  ") : "halt";
  }

  function renderNetwork() {
    $("passes").textContent = fmtN(run.passes);
    var win = $("window"), attn = $("attn"), preds = $("preds"), checks = $("checks");
    win.innerHTML = ""; attn.innerHTML = ""; preds.innerHTML = "";
    var ids = run.last ? run.last.ids : run.window();
    var labels = ["left", "middle", "right"];
    ids.forEach(function (id, i) {
      var w = el("div", "wtok"); w.appendChild(el("span", "pos", labels[i])); w.appendChild(el("b", null, engine.vin[id])); win.appendChild(w);
    });
    var f = engine.net.forward(ids);
    f.attention.forEach(function (a, h) {
      var row = el("div", "ahead"); row.appendChild(el("span", null, "head " + (h + 1)));
      var bars = el("div", "abars");
      a.forEach(function (v, j) {
        var bar = el("div", "abar"), fill = el("i"); fill.style.width = Math.round(v * 100) + "%";
        bar.appendChild(fill); bar.appendChild(el("span", null, labels[j] + " " + Math.round(v * 100) + "%")); bars.appendChild(bar);
      });
      row.appendChild(bars); attn.appendChild(row);
    });
    var lg = f.logits, m = -Infinity, i2;
    for (i2 = 0; i2 < lg.length; i2++) if (lg[i2] > m) m = lg[i2];
    var z = 0, ps = new Float64Array(lg.length);
    for (i2 = 0; i2 < lg.length; i2++) { ps[i2] = Math.exp(lg[i2] - m); z += ps[i2]; }
    var order = Array.from(ps.keys()).sort(function (a, b) { return ps[b] - ps[a]; }).slice(0, 3);
    order.forEach(function (k, r) {
      var row = el("div", "pred" + (r === 0 ? " top" : ""));
      row.appendChild(el("span", null, show(D.vout[k])));
      var pb = el("div", "pbar"), fill = el("i"); fill.style.width = Math.max(1, ps[k] / z * 100) + "%"; pb.appendChild(fill);
      row.appendChild(pb);
      var pct = ps[k] / z * 100;
      row.appendChild(el("span", "pp", pct > 99.95 ? "100%" : pct < 0.05 ? "<0.1%" : pct.toFixed(1) + "%"));
      preds.appendChild(row);
    });
    var app = el("div", "appended");
    app.appendChild(document.createTextNode(run.last ? (run.last.out.length ? "appended " : "") : "next, it will read this window"));
    if (run.last) app.appendChild(el("b", null, run.last.out.length ? run.last.out.map(function (id) { return engine.vin[id]; }).join(" ") : "halt: no step left to take"));
    preds.appendChild(app);
    checks.innerHTML = "";
    [["Different steps used", fmtN(run.seen.size), false],
     ["Never seen in training", fmtN(run.unseen), run.unseen > 0],
     ["Disagreed with the exact rule", fmtN(run.disagree), run.disagree > 0],
     ["Universal-machine rules", String(T.rulesCount), false]].forEach(function (c) {
      var box = el("div", "check" + (c[2] ? " warn" : "")); box.appendChild(el("span", null, c[0])); box.appendChild(el("b", null, c[1]));
      checks.appendChild(box);
    });
  }

  // ---------------------------------------------------------------- wiring

  $("play").addEventListener("click", function () { if (playing) { stop(); render(); } else play(); });
  [["step-pass", "pass"], ["step-utm", "utm"], ["step-sim", "sim"]].forEach(function (p) {
    $(p[0]).addEventListener("click", function () { stop(); advance(p[1]); render(); });
  });
  $("to-end").addEventListener("click", function () {
    stop();
    var guard = 0;
    while (run && !run.verdict && guard++ < 200) advance("fast");
    render();
  });
  $("reset").addEventListener("click", reset);
  $("input").addEventListener("keydown", function (e) { if (e.key === "Enter") reset(); });
  $("input").addEventListener("change", reset);
  var debounce = null;
  $("rules").addEventListener("input", function () { clearTimeout(debounce); debounce = setTimeout(reset, 400); });
  document.querySelector(".theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tct-theme", root.dataset.theme); } catch (e) {}
  });
  $("n-machines").textContent = fmtN(D.training.random_machines);
  buildChips();
  choose("increment");
})();
