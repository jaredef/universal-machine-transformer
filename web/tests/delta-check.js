// The delta run (core.js DeltaRun) against the full lagging-window decoder (FastRun): for every showcase program and
// curve, and random machines, on both universal machines and their networks, the universal machine's whole ring at
// every described step must be the same, and so must the verdict, the forward passes, and the windows the network read
// (distinct, never in training, disagreeing with the exact rule).
//   node delta-check.js        (LIM=400 for a quicker check, LIM=4000 for a longer one (hours: the full decoder is slow), VERBOSE=1 for every program)
const W = __dirname + "/../";
global.window = {}; require(W + "data.js"); require(W + "data2.js");
const T = require(W + "core.js"), PR = require(W + "programs.js"), CV = require(W + "curves.js");
const E = { 1: new T.Engine(window.TCT_DATA), 2: new T.Engine(window.TCT_DATA2) };
const parse = T.Run.prototype.parse;
const LIM = +(process.env.LIM || 1000), VERBOSE = !!process.env.VERBOSE;   // described steps per run at most

function rings(Kind, v, m, tape) {
  const out = [];
  const run = new Kind(E[v], m, tape, x => out.push(x.cells ? x : null));
  if (Kind === T.FastRun) {
    run.onStep = c => out.push(c.state + "@" + c.head + "|" + c.cells.join(" "));
    run.onHalt = c => out.push("halt|" + c.simTape.join("").replace(/_+$/, ""));
  } else {
    run.onStep = r => { if (!r.verdict) { const c = parse.call(r, r.tokens()); out.push(c.state + "@" + c.head + "|" + c.cells.join(" ")); } };
    run.onHalt = r => out.push("halt|" + r.simTape().join("").replace(/_+$/, ""));
  }
  const t0 = Date.now(), lim = LIM;
  while (!run.verdict && run.simSteps < lim) run.run(Kind === T.FastRun ? 1e6 : 2e4);
  return { out, run, ms: Date.now() - t0 };
}
function random(rng, n) {
  const states = Array.from({ length: n }, (_, i) => "s" + i), rules = [];
  for (const q of states) for (const s of "01_") {
    if (rng() < 0.12) continue;
    const nxt = rng() < 0.15 ? (rng() < 0.7 ? "qa" : "qr") : states[Math.floor(rng() * n)];
    rules.push(`${q} ${s} -> ${nxt} ${"01_"[Math.floor(rng() * 3)]} ${"LRRS"[Math.floor(rng() * 4)]}`);
  }
  return { rules, tape: Array.from({ length: 1 + Math.floor(rng() * 6) }, () => "01_"[Math.floor(rng() * 3)]).join("") };
}
let seed = 9; const rng = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const jobs = PR.programs.filter(p => p.id !== "fib").map(p => [p.title, p.rules, p.start, p.input(p.param.value)])
  .concat(CV.curves.map(c => [c.title, c.rules, c.start, "_" + "0".repeat(6)]));
for (let i = 0; i < 40; i++) { const r = random(rng, 1 + Math.floor(rng() * 12)); jobs.push(["random " + i, r.rules, "s0", r.tape]); }
let bad = 0, fullMs = 0, deltaMs = 0, read = 0, passes = 0;
for (const [name, rules, start, tape] of jobs) for (const v of [1, 2]) {
  const m = T.parseMachine(start, rules);
  if (m.order.length > E[v].utm.maxStates) continue;
  const a = rings(T.FastRun, v, m, tape), b = rings(T.DeltaRun, v, m, tape);
  if (!a.run.verdict) { a.out.length = b.out.length = Math.min(a.out.length, b.out.length, LIM); }   // still running at the limit
  const same = a.out.length === b.out.length && a.out.every((x, i) => x === b.out[i]);
  const keys = ["verdict", "passes", "simSteps", "distinct", "unseen", "disagree"];
  const diff = keys.filter(k => a.run.verdict && a.run[k] !== b.run[k]);
  fullMs += a.ms; deltaMs += b.ms; read += b.run.read; passes += b.run.passes;
  if (VERBOSE) console.log(`${name} (version ${v}): ${b.out.length} steps, ${a.run.verdict || "running"}, full ${a.ms} ms, delta ${b.ms} ms`);
  if (!same || diff.length) {
    bad++;
    const at = a.out.findIndex((x, i) => x !== b.out[i]);
    console.log(`MISMATCH ${name} (version ${v}): ${a.out.length} vs ${b.out.length} rings, first difference at ${at}; ` + diff.map(k => `${k} ${a.run[k]} vs ${b.run[k]}`).join(", "));
  }
}
console.log(`${jobs.length} programs on both universal machines: ${bad ? bad + " MISMATCHES" : "the delta run is the full decoder, ring for ring"}`);
console.log(`the network read ${read.toLocaleString()} windows for ${passes.toLocaleString()} forward passes (1 in ${Math.round(passes / read).toLocaleString()}); ` +
  `full decoder ${(fullMs / 1000).toFixed(1)}s, delta ${(deltaMs / 1000).toFixed(1)}s`);
process.exit(bad ? 1 : 0);
