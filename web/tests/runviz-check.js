// Checks the runs umt.html offers under "Run it on the network": every program, with its default input, run by the
// version-2 network through the delta decoder (core.js DeltaRun), read off the transcript one step at a time, must match
// a direct run of the rules configuration for configuration, and each drawing must render.  node runviz-check.js [all]
global.window = global;
require("../data2.js");
const T = require("../core.js"), PR = require("../programs.js"), CU = require("../curves.js"), P = require("../primes.js");
global.TCT_PROGRAMS = PR; global.TCT_CURVES = CU; global.TCT_PRIMES = P;
const RV = require("../runviz.js"), e = new T.Engine(window.TCT_DATA2);
const items = { counter: PR.programs[1], sort: PR.programs[2], add: PR.programs[3], count: PR.programs[4], collatz: PR.programs[0], fib: PR.programs[5],
  dragon: CU.curves[0], koch: CU.curves[1], fibword: CU.curves[2], bb4: null, sieve: null };
let bad = 0;
for (const id of Object.keys(RV.SPECS)) {
  const sp = RV.SPECS[id], param = sp.param.value, it = { src: items[id] || {} };
  const m = RV.setup(id, it, param), want = RV.direct(m.start, m.rules, m.tape);
  const got = [want[0]], t0 = Date.now();
  const r = new T.DeltaRun(e, T.parseMachine(m.start, m.rules), m.tape, (run) => {
    got.push({ state: run.verdict ? "qa" : run.simState(), head: run.simHead(), tape: run.simTape().join("").replace(/_+$/, "") });
  });
  while (!r.verdict) r.run(1e7);
  let ok = 0;
  got.forEach((g, i) => { const w = want[i]; if (w && g.state === w.state && g.tape === w.tape && (w.final || g.head === w.head)) ok++; });
  const d = sp.draw(got, param, true, want.length, sp.finalOf ? sp.finalOf(want) : null);
  const pass = ok === want.length && got.length === want.length && r.disagree === 0 && /<svg/.test(d.svg);
  if (!pass) bad++;
  console.log((pass ? "ok  " : "BAD ") + id.padEnd(8), "steps", want.length - 1, "matched", ok + "/" + want.length, "passes", r.passes, "read", r.read, "wrong", r.disagree, "unseen", r.unseen, Date.now() - t0 + " ms", "|", d.big);
}
process.exit(bad ? 1 : 0);
