// Run every program in web/programs.js on lag3's trained network (through the universal machine, on the lagging-window
// decoder) and check it against the program run directly, configuration by configuration.
//   node programs-check.js [id=param ...]      e.g. node programs-check.js collatz=27 sort=110100
const W = __dirname + "/../";
global.window = {}; require(W + "data.js");
const T = require(W + "core.js"), PR = require(W + "programs.js");
const engine = new T.Engine(window.TCT_DATA);

function direct(rules, start, tape) {                       // one-way tape; a missing rule rejects
  const m = {}; rules.forEach(r => { const [q, s, , n, w, mv] = r.split(" "); m[q + " " + s] = [n, w, mv]; });
  const t = tape.split(""); let h = 0, q = start; const out = [{ state: q, head: h, tape: t.join("").replace(/_+$/, "") }];
  while (q !== "qa" && q !== "qr") {
    const r = m[q + " " + (t[h] || "_")]; if (!r) { q = "qr"; break; }
    t[h] = r[1]; h = Math.max(0, h + (r[2] === "L" ? -1 : r[2] === "R" ? 1 : 0)); q = r[0];
    out.push({ state: q, head: h, tape: t.join("").replace(/_+$/, "") });
  }
  return out;
}

const args = Object.fromEntries(process.argv.slice(2).map(a => a.split("=")));
let allOk = true;
for (const p of PR.programs) {
  if (Object.keys(args).length && !(p.id in args)) continue;
  const param = p.id in args ? (typeof p.param.value === "number" ? +args[p.id] : args[p.id]) : p.param.value;
  const tape = p.input(param), want = direct(p.rules, p.start, tape);
  const got = [{ state: p.start, head: 0, tape: tape.replace(/_+$/, "") }];
  const run = new T.FastRun(engine, T.parseMachine(p.start, p.rules), tape, cfg => got.push({ state: cfg.simState, head: cfg.simHead, tape: cfg.simTape.join("").replace(/_+$/, "") }));
  run.onHalt = cfg => got.push({ state: run.verdict === "ACCEPT" ? "qa" : "qr", head: null, tape: cfg.simTape.join("").replace(/_+$/, ""), final: true });
  const t0 = Date.now();
  while (!run.verdict) run.run(50_000_000);
  let match = got.length === want.length;
  for (let i = 0; match && i < want.length; i++) {
    const g = got[i], w = want[i];
    match = g.state === w.state && g.tape === w.tape && (g.final || g.head === w.head);
  }
  const values = want.map(c => p.value(c)).filter(v => v !== null);
  const expectOk = !p.expect || JSON.stringify(values) === JSON.stringify(p.expect(param));
  const ok = match && expectOk && run.unseen === 0 && run.disagree === 0;
  allOk = allOk && ok;
  console.log(`${ok ? "OK    " : "FAILED"} ${p.id}(${param}): ${want.length - 1} machine steps, every configuration ${match ? "matches" : "DIFFERS"}` +
    `${p.expect ? (expectOk ? ", values as expected" : ", VALUES DIFFER") : ""}; ${run.passes.toLocaleString()} forward passes, ` +
    `${run.utmSteps.toLocaleString()} universal steps, ${run.distinct} distinct steps, ${run.unseen} never in training, ` +
    `${run.disagree} disagreeing; ${((Date.now() - t0) / 1000).toFixed(1)}s; result: ${p.result(want[want.length - 1].tape)}; ` +
    `${values.length} values, last ${JSON.stringify(values.slice(-3))}`);
}
process.exit(allOk ? 0 : 1);
