// The sieve of Eratosthenes (primes.js), run by the 23,844-parameter network through universal machine version 2, on
// the delta decoder (core.js DeltaRun). The tape is drawn as an Ulam spiral, read off the network's transcript after
// every step: what is crossed out, the current prime, the head. The program is also run directly, one step behind,
// only to check each of the network's steps.
(function () {
  "use strict";
  var T = window.TCT, P = window.TCT_PRIMES;
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  var engine = new T.Engine(window.TCT_DATA2);

  var SPEEDS = {
    fast: { label: "as fast as possible" },
    s2000: { label: "2,000 steps a second", rate: 2000 },
    s200: { label: "200 steps a second", rate: 200 },
    s20: { label: "20 steps a second", rate: 20 },
    s3: { label: "3 steps a second", rate: 3 }
  };

  var side, N, prog, machine, run, cells, by, head, k, playing = false, timer = null, elapsed = 0, owed = 0, lastT = 0;
  var dm, dt, dh, dq, matched, mismatched, xy, shown;

  function reset() {
    stop();
    side = +$("size").value; N = side * side - 1;
    prog = P.sieve(N); machine = T.parseMachine(prog.start, prog.rules);
    cells = prog.tape.split(""); by = new Int32Array(N + 1); head = 0; k = 0;
    dm = {}; prog.rules.forEach(function (r) { var a = r.split(" "); dm[a[0] + " " + a[1]] = a; });
    shown = prog.start; dt = prog.tape.split(""); dh = 0; dq = prog.start; matched = 0; mismatched = 0; elapsed = 0;
    run = new T.DeltaRun(engine, machine, prog.tape, onStep);
    xy = spiral(side);
    var rl = $("rules"); rl.innerHTML = ""; prog.rules.forEach(function (r) { rl.appendChild(el("div", "rule", r)); });
    $("nstates").textContent = machine.order.length; $("nrules").textContent = prog.rules.length.toLocaleString();
    $("play").textContent = "▶ Sieve";
    render();
  }

  // after every step of the program: the cell it wrote and where its head is now, read off the network's transcript;
  // then the same step taken directly, to check it
  function onStep(r) {
    var h = r.simHead(), w = r.simCell(head);
    if (w === "0" && cells[head] !== "0" && head <= N) by[head] = k;
    if (w === "_" && head >= 2 && head <= N) k = head;     // a prime is marked
    cells[head] = w; head = h;
    var rule = dm[dq + " " + (dt[dh] || "_")], ok = false;
    if (rule) {
      var at = dh;
      dt[at] = rule[4]; dh = Math.max(0, dh + (rule[5] === "L" ? -1 : rule[5] === "R" ? 1 : 0)); dq = rule[3];
      ok = w === rule[4] && h === dh;
    }
    if (ok) matched++; else if (!r.verdict || rule) mismatched++;
  }

  // ---------------------------------------------------------------- running, at the chosen speed
  function tick(now) {
    if (!playing) return;
    var sp = SPEEDS[$("speed").value], t0 = performance.now();
    if (!sp.rate) { do run.run(200000); while (!run.verdict && performance.now() - t0 < 30); }
    else {
      owed += Math.min(0.25, (now - lastT) / 1000) * sp.rate; lastT = now;
      run.stopEachStep = true;
      while (owed >= 1 && !run.verdict && performance.now() - t0 < 30) { run.run(1e7); owed--; }
      run.stopEachStep = false;
    }
    elapsed += performance.now() - t0;
    render();
    if (run.verdict) { stop(); return; }
    timer = requestAnimationFrame(tick);
  }
  function stop() { playing = false; cancelAnimationFrame(timer); $("play").textContent = run && run.verdict ? "▶ Sieve again" : "▶ Sieve"; }

  // ---------------------------------------------------------------- the numbers
  function proven(n) { return cells[n] === "1" && (run.verdict || n < k * k) || (n === k && n >= 2); }
  function render() {
    var count = 0;
    for (var n = 2; n <= N; n++) if (proven(n)) count++;
    $("found").textContent = count.toLocaleString() + " / " + P.primesUpto(N).length.toLocaleString();
    $("passes").textContent = run.passes.toLocaleString();
    var fast = $("speed").value === "fast";
    $("rate").textContent = fast && elapsed > 300 ? Math.round(run.passes / elapsed / 1e6).toLocaleString() + " billion per second, " +
      Math.round(run.read / elapsed / 1000).toLocaleString() + " million read by the network" : "";
    $("sim").textContent = run.simSteps.toLocaleString();
    $("utm").textContent = run.utmSteps.toLocaleString();
    $("distinct").textContent = run.distinct.toLocaleString();
    $("unseen").textContent = run.unseen; $("c-unseen").classList.toggle("warn", run.unseen > 0);
    $("disagree").textContent = run.disagree; $("c-dis").classList.toggle("warn", run.disagree > 0);
    $("match").textContent = matched.toLocaleString() + " / " + (matched + mismatched).toLocaleString();
    $("c-match").classList.toggle("warn", mismatched > 0);
    var q = run.verdict ? null : (shown = run.simState() || shown);
    $("in-state").textContent = run.verdict ? "halted" : q;
    $("in-say").textContent = run.verdict ? "" : say(q);
    var v = $("verdict");
    if (run.verdict) {
      var got = []; cells.forEach(function (c, i) { if (c === "1" && i <= N) got.push(i); });
      var ok = run.verdict === "ACCEPT" && mismatched === 0 && run.disagree === 0 && JSON.stringify(got) === JSON.stringify(prog.primes);
      v.textContent = ok ? "Done: all " + got.length + " primes up to " + N.toLocaleString() + ", found by the 23,844-parameter network running a " +
        machine.order.length + "-state program, every one of its " + run.simSteps.toLocaleString() + " steps matching the program run directly, from " +
        run.passes.toLocaleString() + " forward passes."
        : "This run differs from the program run directly; see the checks below.";
      v.className = "verdict2 " + (ok ? "ok" : "bad");
    } else { v.textContent = ""; v.className = "verdict2"; }
    draw();
  }
  function say(q) {
    var m;
    if ((m = /^k(\d+)$/.exec(q))) return "· is " + m[1] + " still lit? If so it is prime";
    if ((m = /^m(\d+)_(\d+)$/.exec(q))) return "· crossing out multiples of " + m[1] + ": the next is " + (+m[2] === 0 ? "this cell" : m[2] + " cell" + (+m[2] === 1 ? "" : "s") + " on");
    if ((m = /^r(\d+)$/.exec(q))) return "· walking back to " + m[1];
    return q === "s" || q === "s1" ? "· starting" : "";
  }

  // ---------------------------------------------------------------- the spiral
  function spiral(s) {                                     // n -> [column, row], 1 in the middle, counterclockwise
    var pos = new Int16Array(2 * (s * s + 1)), x = (s - 1) >> 1, y = (s - 1) >> 1, n = 1, len = 1, dir = 0;
    var DX = [1, 0, -1, 0], DY = [0, -1, 0, 1];
    pos[2] = x; pos[3] = y;
    while (n < s * s) {
      for (var t = 0; t < 2; t++) {
        for (var i = 0; i < len && n < s * s; i++) { x += DX[dir]; y += DY[dir]; n++; pos[2 * n] = x; pos[2 * n + 1] = y; }
        dir = (dir + 1) % 4;
      }
      len++;
    }
    return pos;
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function draw() {
    var cv = $("spiral"), dpr = window.devicePixelRatio || 1, w = cv.clientWidth;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(w * dpr); }
    var ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, w);
    var c = w / side, g = c > 8 ? 1.5 : c > 5 ? 1 : 0.5, r = Math.min(3, c * 0.18);
    var ink = css("--ink"), accent = css("--accent"), accent2 = css("--accent-2");
    for (var n = 1; n <= N; n++) {
      var X = xy[2 * n] * c, Y = xy[2 * n + 1] * c, s = cells[n], fill, a = 1;
      if (n === k && !run.verdict) fill = accent2;
      else if (proven(n)) fill = accent;
      else if (s === "1") { fill = ink; a = 0.38; }
      else if (by[n] && by[n] === k && !run.verdict) { fill = accent2; a = 0.55; }
      else { fill = ink; a = 0.08; }
      ctx.globalAlpha = a; ctx.fillStyle = fill;
      ctx.beginPath(); ctx.roundRect(X + g / 2, Y + g / 2, c - g, c - g, r); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (!run.verdict && head >= 1 && head <= N) {          // the head: an outline on its cell
      ctx.strokeStyle = ink; ctx.lineWidth = Math.max(1.5, c * 0.12);
      ctx.strokeRect(xy[2 * head] * c + ctx.lineWidth / 2, xy[2 * head + 1] * c + ctx.lineWidth / 2, c - ctx.lineWidth, c - ctx.lineWidth);
    }
    if (side < 16) {                                       // small spirals: the numbers themselves
      ctx.font = "600 " + Math.round(c * 0.32) + "px " + css("--mono"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
      for (n = 1; n <= N; n++) {
        ctx.fillStyle = proven(n) || (n === k && !run.verdict) ? css("--panel") : css("--muted");
        ctx.fillText(n, xy[2 * n] * c + c / 2, xy[2 * n + 1] * c + c / 2);
      }
    }
  }

  // ---------------------------------------------------------------- controls
  Object.keys(SPEEDS).forEach(function (s) { $("speed").appendChild(new Option(SPEEDS[s].label, s)); });
  P.sizes.forEach(function (z) { $("size").appendChild(new Option(z[1] + " (" + z[2] + ")", z[0])); });
  $("play").addEventListener("click", function () {
    if (playing) { stop(); return; }
    if (run.verdict) reset();
    playing = true; $("play").textContent = "❚❚ Pause"; lastT = performance.now(); owed = 1; timer = requestAnimationFrame(tick);
  });
  $("step").addEventListener("click", function () {
    if (run.verdict) reset();
    stop(); run.stopEachStep = true; var s = run.simSteps; while (!run.verdict && run.simSteps === s) run.run(1e7); run.stopEachStep = false; render();
  });
  $("reset").addEventListener("click", reset);
  $("size").addEventListener("change", reset);
  window.addEventListener("resize", draw);
  document.querySelector(".theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tct-theme", root.dataset.theme); } catch (e) {}
    draw();
  });
  var q = new URLSearchParams(location.search);            // primes.html?n=23&speed=s200&play
  if (SPEEDS[q.get("speed")]) $("speed").value = q.get("speed");
  $("size").value = String(P.sizes[1][0]);
  if (q.has("n") && P.sizes.some(function (z) { return String(z[0]) === q.get("n"); })) $("size").value = q.get("n");
  reset();
  if (q.has("play")) $("play").click();
})();
