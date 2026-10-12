// The busy beaver: lag3's trained network runs the universal machine, which runs a busy beaver champion.
(function () {
  "use strict";
  var D = window.TCT_DATA, T = window.TCT, engine = new T.Engine(D);
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  // The proven champions (blank is _), each after a prelude w that walks right over `pad` 0s, so the champion has room
  // on the left of the universal machine's one-way tape. Same machines as machines/lag5/derived/bb.py.
  var CHAMPS = {
    bb4: { states: 4, steps: 107, ones: 13, pad: 10,
           rules: ["A _ -> B 1 R", "A 1 -> B 1 L", "B _ -> A 1 L", "B 1 -> C _ L",
                   "C _ -> qa 1 R", "C 1 -> D 1 L", "D _ -> D 1 R", "D 1 -> A _ R"] },
    bb2: { states: 2, steps: 6, ones: 4, pad: 2,
           rules: ["A _ -> B 1 R", "A 1 -> B 1 L", "B _ -> A 1 L", "B 1 -> qa 1 R"] }
  };
  var PRELUDE = ["w 0 -> w _ R", "w _ -> A _ S"];

  function table(lines) {
    var m = {};
    lines.forEach(function (r) { var p = r.split(" "); m[p[0] + " " + p[1]] = { next: p[3], write: p[4], move: p[5] }; });
    return m;
  }
  function direct(lines, tape) {                           // the champion run directly, on the same one-way tape
    var m = table(lines), t = tape.split(""), h = 0, q = "w", out = [{ state: q, head: h, tape: trim(t) }];
    while (q !== "qa") {
      var r = m[q + " " + (t[h] || "_")];
      t[h] = r.write; h = Math.max(0, h + (r.move === "L" ? -1 : r.move === "R" ? 1 : 0)); q = r.next;
      out.push({ state: q, head: h, tape: trim(t) });
    }
    return out;
  }
  function trim(t) { return (Array.isArray(t) ? t.join("") : t).replace(/_+$/, ""); }
  function ones(s) { return (s.match(/1/g) || []).length; }

  var champ, lines, rules, want, run, got, playing = false, timer = null, width = 0;

  function reset() {
    stop();
    var c = CHAMPS[champ];
    lines = PRELUDE.concat(c.rules); rules = table(lines);
    var tape = new Array(c.pad + 1).join("0");
    want = direct(lines, tape);
    width = want.reduce(function (w, s) { return Math.max(w, s.tape.length, s.head + 1); }, 0);
    run = new T.FastRun(engine, T.parseMachine("w", lines), tape, function (cfg, r) {
      got.push({ state: cfg.simState, head: cfg.simHead, tape: trim(cfg.simTape), at: r.at });
    });
    run.stopEachStep = true;
    run.onHalt = function (cfg, r) { got.push({ state: "qa", head: cfg.simHead, tape: trim(cfg.simTape), at: r.at, final: true }); };
    got = [{ state: "w", head: 0, tape: tape, at: 0 }];
    document.querySelectorAll("#champs .chip").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.champ === champ); });
    $("steps-of").textContent = "of the record " + c.steps;
    $("ones-of").textContent = "of the record " + c.ones;
    $("encoded").textContent = run.encoded;
    buildRules();
    $("play").textContent = "▶ Run";
    render();
  }

  function buildRules() {
    var t = $("rules"); t.innerHTML = "";
    var head = el("tr"); head.appendChild(el("th", null, "state"));
    ["0", "_", "1"].forEach(function (s) { head.appendChild(el("th", null, "reads " + s)); });
    var thead = el("thead"); thead.appendChild(head); t.appendChild(thead);
    var tb = el("tbody");
    ["w", "A", "B", "C", "D"].forEach(function (q) {
      if (q !== "w" && !rules[q + " _"]) return;
      var tr = el("tr", q === "w" ? "prelude" : null); tr.appendChild(el("th", null, q));
      ["0", "_", "1"].forEach(function (s) {
        var r = rules[q + " " + s], td = el("td", null, r ? r.write + " " + r.move + " " + (r.next === "qa" ? "halt" : r.next) : "");
        td.dataset.key = q + " " + s; tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb);
  }

  function step() {                                         // one described step (or the halt)
    if (run.verdict) return false;
    while (!run.verdict) {
      var before = got.length;
      run.run(50000000);
      if (got.length > before) break;
    }
    return true;
  }

  function tick() {
    if (!playing) return;
    var ms = +$("speed").value;
    if (ms === 0) {                                          // as fast as possible: ~30 ms of work per frame
      var t0 = performance.now();
      while (!run.verdict && performance.now() - t0 < 30) step();
    } else step();
    render();
    if (run.verdict) { stop(); return; }
    timer = ms === 0 ? requestAnimationFrame(tick) : setTimeout(tick, ms);
  }
  function stop() {
    playing = false; clearTimeout(timer); cancelAnimationFrame(timer);
    $("play").textContent = run && run.verdict ? "▶ Run again" : "▶ Run";
  }

  function render() {
    var c = CHAMPS[champ], n = got.length - 1, cur = got[n], prev = got[n - 1];
    var champSteps = Math.max(0, n - c.pad - 1);
    $("steps").textContent = champSteps;
    $("ones").textContent = ones(cur.tape);
    $("passes").textContent = (cur.at || 0).toLocaleString();
    $("this-step").textContent = prev ? (cur.at - prev.at).toLocaleString() + " for the last step" : "";
    $("utm").textContent = run.utmSteps.toLocaleString();
    $("distinct").textContent = run.distinct.toLocaleString();
    $("unseen").textContent = run.unseen; $("c-unseen").classList.toggle("warn", run.unseen > 0);
    $("disagree").textContent = run.disagree; $("c-dis").classList.toggle("warn", run.disagree > 0);
    var matched = 0;
    for (var i = 0; i < got.length; i++) {
      var g = got[i], w = want[i];
      if (w && g.state === w.state && g.tape === w.tape && (g.final || g.head === w.head)) matched++;
    }
    $("match").textContent = matched + " / " + got.length;
    $("c-match").classList.toggle("warn", matched < got.length);
    $("phase").textContent = n === 0 ? "Its tape, its head, and its state, read off the universal machine's tape after every step."
      : n <= c.pad + 1 ? "The prelude: walking right to make room on the left (step " + n + " of " + (c.pad + 1) + ")."
      : cur.state === "qa" ? "Halted." : "Running the champion: state " + cur.state + ".";

    var tape = $("tape"); tape.innerHTML = "";
    for (var j = 0; j < width; j++) {
      var s = cur.tape[j] || "_", cell = el("div", "bbcell" + (s === "1" ? " one" : s === "0" ? " zero" : ""), s === "0" ? "0" : "");
      if (j === cur.head && !cur.final) { cell.classList.add("head"); cell.appendChild(el("span", "badge", cur.state)); }
      tape.appendChild(cell);
    }
    document.querySelectorAll("#rules td").forEach(function (td) { td.classList.remove("on"); });
    if (prev) {
      var key = prev.state + " " + (prev.tape[prev.head] || "_"), r = rules[key];
      var td = document.querySelector('#rules td[data-key="' + key + '"]'); if (td) td.classList.add("on");
      $("used").textContent = "State " + prev.state + " read " + (prev.tape[prev.head] || "_") + ": wrote " + r.write +
        ", moved " + { L: "left", R: "right", S: "nowhere" }[r.move] + ", " + (r.next === "qa" ? "and halted." : "went to " + r.next + ".");
    } else $("used").innerHTML = "&nbsp;";

    var v = $("verdict");
    if (run.verdict) {
      var ok = matched === got.length && got.length === want.length && run.unseen === 0 && run.disagree === 0;
      v.textContent = ok ? "Halted after " + champSteps + " steps with " + ones(cur.tape) + " ones: the record for " + c.states +
        " states, reproduced by the network through every configuration, from " + cur.at.toLocaleString() + " forward passes."
        : "The run differs from the champion run directly; see the checks above.";
      v.className = "verdict2 " + (ok ? "ok" : "bad");
    } else { v.textContent = ""; v.className = "verdict2"; }
    ribbon();
  }

  function ribbon() {
    var cv = $("ribbon"), wrap = $("ribbon-wrap"), cols = want.length, rows = width;
    var cell = Math.max(5, Math.min(22, Math.floor((wrap.clientWidth - 2) / cols)));
    var w = cell * cols, h = cell * rows, dpr = window.devicePixelRatio || 1;
    if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; cv.style.width = w + "px"; cv.style.height = h + "px"; }
    var ctx = cv.getContext("2d"), cs = getComputedStyle(document.documentElement);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = cs.getPropertyValue("--soft").trim(); ctx.fillRect(0, 0, w, h);
    var ink = cs.getPropertyValue("--ink").trim(), muted = cs.getPropertyValue("--rule").trim(), acc = cs.getPropertyValue("--accent").trim();
    var gap = cell > 6 ? 1 : 0;
    for (var t = 0; t < got.length; t++) {
      var g = got[t];
      for (var j = 0; j < rows; j++) {
        var s = g.tape[j];
        if (s === "1" || s === "0") { ctx.fillStyle = s === "1" ? ink : muted; ctx.fillRect(t * cell, j * cell, cell - gap, cell - gap); }
      }
      if (!g.final) {
        ctx.strokeStyle = acc; ctx.lineWidth = Math.max(1.5, cell / 5);
        ctx.strokeRect(t * cell + ctx.lineWidth / 2, g.head * cell + ctx.lineWidth / 2, cell - gap - ctx.lineWidth, cell - gap - ctx.lineWidth);
      }
    }
    if (playing) wrap.scrollLeft = Math.max(0, got.length * cell - wrap.clientWidth + cell * 4);
  }

  $("champs").addEventListener("click", function (e) {
    var b = e.target.closest(".chip"); if (!b) return; champ = b.dataset.champ; reset();
  });
  $("play").addEventListener("click", function () {
    if (playing) { stop(); return; }
    if (run.verdict) reset();
    playing = true; $("play").textContent = "❚❚ Pause"; tick();
  });
  $("step").addEventListener("click", function () { if (run.verdict) reset(); stop(); step(); render(); });
  $("reset").addEventListener("click", reset);
  window.addEventListener("resize", ribbon);
  document.querySelector(".theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tct-theme", root.dataset.theme); } catch (e) {}
    ribbon();
  });

  var q = new URLSearchParams(location.search);               // beaver.html?bb=2&speed=0&play
  champ = q.get("bb") === "2" ? "bb2" : "bb4";
  if (["600", "140", "0"].indexOf(q.get("speed")) >= 0) $("speed").value = q.get("speed");
  reset();
  if (q.has("play")) $("play").click();
})();
