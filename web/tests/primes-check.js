// The sieve of Eratosthenes (primes.js) on the 23,844-parameter network, universal machine version 2, through the delta
// decoder (DeltaRun, checked against the full decoder by delta-check.js): at every size the page offers, every step must
// match the program run directly, and the primes left on the tape must be the primes.
//   node primes-check.js [sides, e.g. 11,23]
const W = __dirname + "/../";
global.window = {}; require(W + "data2.js");
const T = require(W + "core.js"), P = require(W + "primes.js");
const E = new T.Engine(window.TCT_DATA2);
const sides = process.argv[2] ? process.argv[2].split(",").map(Number) : P.sizes.map(z => z[0]);
let ok = true;
console.log("| Numbers | States | Rules | Steps | Universal steps | Forward passes | Windows read by the network | Never in training | Disagreeing | Primes right | Seconds |\n|---|---|---|---|---|---|---|---|---|---|---|");
for (const side of sides) {
  const N = side * side - 1, s = P.sieve(N), m = T.parseMachine(s.start, s.rules);
  const dm = {}; s.rules.forEach(r => { const a = r.split(" "); dm[a[0] + " " + a[1]] = a; });
  const dt = s.tape.split(""); let dh = 0, dq = s.start, head = 0, bad = 0;
  const run = new T.DeltaRun(E, m, s.tape, r => {          // each step, read off the network's transcript, against the program run directly
    const h = r.simHead(), w = r.simCell(head), rule = dm[dq + " " + (dt[dh] || "_")];
    if (!rule) { bad++; return; }
    dt[dh] = rule[4]; dh = Math.max(0, dh + (rule[5] === "L" ? -1 : rule[5] === "R" ? 1 : 0)); dq = rule[3];
    if (w !== rule[4] || h !== dh) bad++;
    head = h;
  });
  const t0 = Date.now();
  while (!run.verdict) run.run(1e6);
  const got = []; run.simTape().forEach((c, i) => { if (c === "1") got.push(i); });
  const right = JSON.stringify(got) === JSON.stringify(s.primes) && run.verdict === "ACCEPT" && bad === 0 && run.disagree === 0;
  ok = ok && right;
  console.log(`| 1 to ${N.toLocaleString()} | ${m.order.length} | ${s.rules.length} | ${run.simSteps.toLocaleString()} | ${run.utmSteps.toLocaleString()} | ${run.passes.toLocaleString()} | ` +
    `${run.read.toLocaleString()} | ${run.unseen} (of ${run.distinct.toLocaleString()}) | ${run.disagree} | ${right ? "yes, " + got.length : "NO"} | ${((Date.now() - t0) / 1000).toFixed(0)} |`);
}
process.exit(ok ? 0 : 1);
