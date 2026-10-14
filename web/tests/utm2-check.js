// Universal machine version 2 in the browser engine: run the showcase programs and curves through both universal
// machines, each on its own trained network (version 1: the committed lag3 network; version 2: data2.js, 23,844
// parameters), on the lagging-window decoder, check every result against the program run directly, and compare the
// forward passes. Then a program only version 2 is fast enough for: the dragon curve's machine spread among 256 states.
//   node utm2-check.js
const W = __dirname + "/../";
global.window = {}; require(W + "data.js"); require(W + "data2.js");
const T = require(W + "core.js"), PR = require(W + "programs.js"), CV = require(W + "curves.js");
const E = { 1: new T.Engine(window.TCT_DATA), 2: new T.Engine(window.TCT_DATA2) };

function direct(rules, start, tape, emit) {
  const m = {}; rules.forEach(r => { const [q, s, , n, w, mv] = r.split(" "); m[q + " " + s] = [n, w, mv]; });
  const t = tape.split(""); let h = 0, q = start; const vals = [], tapes = [];
  while (q !== "qa" && q !== "qr") { const r = m[q + " " + (t[h] || "_")]; if (!r) { q = "qr"; break; }
    t[h] = r[1]; h = Math.max(0, h + (r[2] === "L" ? -1 : r[2] === "R" ? 1 : 0)); q = r[0];
    tapes.push(q + "|" + t.join("").replace(/_+$/, "")); if (emit) { const v = emit(q); if (v !== null) vals.push(v); } }
  return { tapes, vals, verdict: q };
}
function through(v, rules, start, tape, emit) {
  const tapes = [], vals = [];
  const run = new T.FastRun(E[v], T.parseMachine(start, rules), tape, c => {
    tapes.push(c.simState + "|" + c.simTape.join("").replace(/_+$/, ""));
    if (emit) { const x = emit(c.simState); if (x !== null) vals.push(x); } });
  run.onHalt = c => tapes.push((run.verdict === "ACCEPT" ? "qa" : "qr") + "|" + c.simTape.join("").replace(/_+$/, ""));
  const t0 = Date.now(); while (!run.verdict) run.run(50e6);
  return { tapes, vals, verdict: run.verdict, passes: run.passes, steps: run.simSteps, unseen: run.unseen, disagree: run.disagree, ms: Date.now() - t0 };
}
let ok = true;
const jobs = PR.programs.filter(p => p.id !== "fib").map(p => [p.title, p.rules, p.start, p.input(p.param.value), null])
  .concat(CV.curves.map(c => [c.title, c.rules, c.start, "_" + "0".repeat(8), c.emit]));
console.log("| Program | Steps | Version 1 passes per step | Version 2 passes per step | Speedup | Both correct |\n|---|---|---|---|---|---|");
for (const [name, rules, start, tape, emit] of jobs) {
  const want = direct(rules, start, tape, emit), r = {};
  for (const v of [1, 2]) {
    const g = through(v, rules, start, tape, emit);
    const final = want.tapes.slice(0, -1).every((x, i) => x === g.tapes[i]) && g.tapes.length === want.tapes.length && (!emit || JSON.stringify(g.vals) === JSON.stringify(want.vals));
    r[v] = { g, good: final && g.unseen === 0 && g.disagree === 0 };
    ok = ok && r[v].good;
  }
  const p1 = r[1].g.passes / r[1].g.steps, p2 = r[2].g.passes / r[2].g.steps;
  console.log(`| ${name} | ${r[2].g.steps.toLocaleString()} | ${Math.round(p1).toLocaleString()} | ${Math.round(p2).toLocaleString()} | ${(p1 / p2).toFixed(1)}x | ${r[1].good && r[2].good ? "yes" : "NO (v1 " + r[1].good + ", v2 " + r[2].good + ")"} |`);
}
// a big program: the dragon curve's machine (3-bit counter) with 248 states that are never reached, spread by name
const dragon = CV.curves[0];
const pad = Array.from({ length: 248 }, (_, i) => "z" + ((i * 7919) % 997).toString(36) + i);
const big = dragon.rules.concat(pad.map(z => z + " 0 -> " + z + " 0 R"));
const want = direct(big, "s", "_000", dragon.emit), g = through(2, big, "s", "_000", dragon.emit);
const good = JSON.stringify(g.vals) === JSON.stringify(want.vals) && g.unseen === 0 && g.disagree === 0;
ok = ok && good;
console.log(`\ndragon curve among 256 states, version 2: ${g.steps.toLocaleString()} steps, ${g.vals.length} turns ${good ? "all right" : "WRONG"}, ` +
  `${Math.round(g.passes / g.steps).toLocaleString()} passes per step, ${g.unseen} never in training, ${g.disagree} disagreeing, ${(g.ms / 1000).toFixed(1)}s`);
process.exit(ok ? 0 : 1);
