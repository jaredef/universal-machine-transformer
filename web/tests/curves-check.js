// Run the curves of web/curves.js on both networks (the 8,170-parameter one and lag3) through the universal machine and
// compare every announced value with the machine run directly and with the curve's formula.
//   node curves-check.js [bits]      (default 8: 255 values per curve)
const W = __dirname + "/../";
global.window = {}; require(W + "data.js"); require(W + "net8.js"); require(W + "table.js");
const T = require(W + "core.js"), CV = require(W + "curves.js"), D = window.TCT_DATA, TB = window.TCT_TABLE;
const k = +process.argv[2] || 8;
const train8 = Array.from({ length: TB.labels.length }, (_, i) => [0, 1, 2].map(t => TB.vin[TB.steps[3 * i + t]]).join("\u0001"));
const nets = { "8,170": new T.Engine(Object.assign({}, D, { weights: window.TCT_NET8.weights, heads: window.TCT_NET8.heads, train: train8 })), "211,270": new T.Engine(D) };
let ok = true;
for (const c of CV.curves) for (const [name, e] of Object.entries(nets)) {
  const vals = [], run = new T.FastRun(e, T.parseMachine(c.start, c.rules), "_" + "0".repeat(k), cfg => { const v = c.emit(cfg.simState); if (v !== null) vals.push(v); });
  const t0 = Date.now(); while (!run.verdict) run.run(50e6);
  const good = JSON.stringify(vals) === JSON.stringify(c.expect(k)) && run.unseen === 0 && run.disagree === 0 && run.verdict === "ACCEPT";
  ok = ok && good;
  console.log(`${good ? "OK    " : "FAILED"} ${c.id} k=${k} on ${name} parameters: ${vals.length} values ${good ? "as expected" : "DIFFER"}; ${run.simSteps} machine steps, ${run.passes.toLocaleString()} forward passes, ${run.distinct} distinct steps, ${run.unseen} never in training, ${run.disagree} disagreeing; ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
process.exit(ok ? 0 : 1);
