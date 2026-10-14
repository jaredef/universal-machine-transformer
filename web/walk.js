// walk.html: one forward pass, followed by scrolling. The scene (inside-scene.js) stays in place while the steps scroll
// past; the camera moves between the steps' views with the scroll, and each value appears as the pass reaches it.
import { INSIDE, vin, vout, comma, f3, pct, POS, fmtP, code, describe, maxAbs, windows, analyse, drawStrip, revealStrip,
         budgetTable, CH, ALL, R2, createScene, themeButton } from "./inside-scene.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const src = windows();
const stage = $("stage");
stage.querySelector(".v-msg").remove();
const S = createScene(stage, $("tip"), { controls: false });
let view = null;

const tokens = (v) => [0, 1, 2].map((p) => code(vin[v.ids[p]])).join(", ");
const on = (v) => Array.from(v.f.hidden).filter((x) => x > 0).length;
function heads(v) {
  return [0, 1].map((h) => {
    const a = v.f.attention[h], top = a.indexOf(Math.max(...a));
    return "<span class='hk h" + (h + 1) + "'>Head " + (h + 1) + "</span> puts " + [0, 1, 2].map((p) => "<b>" + pct(a[p]) + "</b> on " + POS[p]).join(", ") +
      ": it looks mostly at " + POS[top] + ".";
  }).join("<br>");
}

// The steps: the stage of the pass each one shows, where the camera looks, and what it says.
const WIDE = [[-6.6, 0, 5.6], [-6.6, 27, 30.5]];         // the whole scene, with room for a tall stage
const STEPS = [
  { ch: 0, view: WIDE, title: "One pass",
    text: (v) => "<p>" + (v.live ? "This is forward pass " + comma(v.pass) + " of the sieve to 120, running live in this page." :
      "This is forward pass " + comma(v.pass) + " of the " + comma(INSIDE.passes) + " the sieve to 120 takes, and the network " +
      "never saw this window in training.") + " It is reading three tokens of the universal machine's transcript:</p>" +
      "<div class='strip' id='strip'></div><ul class='toks'>" + [0, 1, 2].map((p) => "<li><b>" + POS[p] + "</b> " + code(vin[v.ids[p]]) +
      ", " + describe(vin[v.ids[p]]) + "</li>").join("") + "</ul><p>Its job is the token that goes at c's place in the next ring.</p>" +
      "<p class='muted'>In the picture, every bar is one of the network's numbers, all 23,844 of them, laid out in the order the pass uses them.</p>" },
  { ch: 1, view: [[-15.6, 0, 0.3], [-15.6, 8.6, 9.2]], title: "Look up the tokens",
    text: (v) => "<p>To the network, a token is nothing but its row in the embedding table <b>E</b>: 8 numbers for each of " +
      "1,351 tokens. The table is 10,808 parameters, 45% of the network.</p><p>The pass begins by lifting out three rows, " +
      "for " + tokens(v) + ".</p>" },
  { ch: 1, view: [[-8.6, 0, -0.9], [-8.6, 4.6, 4.6]], title: "Where they sit",
    text: (v) => "<p>A position vector from <b>P</b> is added to each row, so the same token reads differently as l, c, or r.</p>" +
      "<p>Three vectors of 8 numbers, <b>x</b>. From here on, that is all the network has to go on.</p>" },
  { ch: 2, view: CH[2].view, title: "Queries, keys and values",
    text: (v) => "<p>Three 8 × 8 matrices. Only r's position asks a question, its <b>query</b> q = Wq·x<sub>r</sub>, the way a " +
      "language model predicts from its last token. Every position answers with a <b>key</b>, k = Wk·x, and offers a " +
      "<b>value</b>, v = Wv·x.</p><p>The 8 numbers are split between two attention heads of 4 each: " +
      "<span class='hk h1'>head 1</span> and <span class='hk h2'>head 2</span>, marked under each vector.</p>" },
  { ch: 3, view: [[-1.6, 0.4, -0.7], [-0.6, 6.3, 6.1]], title: "Attention",
    text: (v) => "<p>Each head compares its part of the query with the same part of each key, a dot product, and turns the " +
      "three scores into weights that add up to 1. The arcs from the query to the keys are as thick as the weights.</p><p>" + heads(v) + "</p>" },
  { ch: 3, view: [[-0.9, 0, 2.5], [-0.9, 4.6, 7.4]], title: "The heads' mix",
    text: (v) => "<p>Each head takes that mix of the three values, its arcs again as thick as the weights. Side by side, the " +
      "two heads' mixes make <b>y</b>, 8 numbers that carry what attention found.</p>" },
  { ch: 4, view: CH[4].view, title: "Back into the stream",
    text: (v) => "<p><b>Wo</b> turns the mix into an update, <b>o</b>, and adds it to r's own vector: h = x<sub>r</sub> + o. That " +
      "running sum is the <b>residual stream</b>, everything the pass knows so far.</p>" +
      (maxAbs(v.f.o) > maxAbs(v.f.x[2]) ? "<p>The update is larger than what it is added to (" + f3(maxAbs(v.f.o)) + " against " +
      f3(maxAbs(v.f.x[2])) + " at most): attention has rewritten r's vector with what it found.</p>" : "") },
  { ch: 5, view: [[-16.6, 0, R2 + 0.2], [-16.6, 6.6, R2 + 6.8]], title: "The MLP",
    text: (v) => "<p>32 units, each a row of <b>W1</b>. A unit takes a weighted sum of the stream, adds a bias, and passes it " +
      "on only if it is positive (ReLU).</p><p>For this window <b>" + on(v) + " of the 32</b> are on, the column of bars beside W1.</p>" },
  { ch: 5, view: [[-14.6, 0, R2 + 1.4], [-14.6, 6, R2 + 7.8]], title: "Back again",
    text: (v) => "<p><b>W2</b> adds what the units found back into the stream. Its largest number goes from " + f3(maxAbs(v.f.h1)) +
      " to " + f3(maxAbs(v.f.h2)) + ".</p><p class='muted'>Attention and the MLP together are 832 parameters. Everything else is the two tables.</p>" },
  { ch: 6, view: [[-1.2, 0, R2], [-1.2, 12, R2 + 13]], title: "Scoring every output",
    text: (v) => "<p>The unembedding <b>U</b> has a row for every output the network can write: 1,351 single tokens, 4 pairs " +
      "(written when the machine's head moves past the end of the ring), and nothing at all (the machine halts).</p><p>Each " +
      "output's score is its row's dot product with the stream, plus a bias, and a softmax turns the scores into " +
      "probabilities, the bars on the right. The highest:</p><ol class='toks'>" + v.order.slice(0, 3).map((i) => "<li>" +
      code(vout[i]) + " " + fmtP(v.prob(i)) + "</li>").join("") + "</ol>" },
  { ch: 7, view: CH[7].view, title: "What it writes",
    text: (v) => {
      const right = v.wrote === v.exact;
      return "<p>It writes " + (v.wrote ? code(v.wrote) : "nothing (it halts)") + ", with probability " + fmtP(v.prob(v.f.best)) +
        ". The universal machine's exact rule says " + (v.exact ? code(v.exact) : "halt") + ": " +
        (right ? "<b class='ok'>the same ✓</b>" : "<b class='bad'>different ✗</b>") + ".</p>" +
        (v.live ? "<p>" + (v.trained ? "This window was in its training data." : "It never saw this window in training.") + "</p>" :
        "<p>It never saw this window in training. The sieve to 120 reads 12,406 different windows, and only 3 were not in " +
        "training. All three come in the last 0.2% of the run, as the universal machine jumps to the next state by its binary " +
        "address, and the network gets all three right.</p>");
    } },
  { ch: ALL, view: WIDE, title: "And again",
    text: (v) => "<p>That was one pass. Then the window moves one token to the right, and the next one begins. The sieve to " +
      "120 takes " + comma(INSIDE.passes) + " of them, and the sieve to 1,680 takes 9.9 trillion. Each step of the program " +
      "comes out right only if every pass does.</p><div class='row buttons'>" +
      "<button class='btn' type='button' data-go='unseen'>Follow a window it never saw</button>" +
      "<button class='btn' type='button' data-go='live'>Follow a pass from the live run</button></div>" +
      "<p class='muted'><a href='inside.html'>Explore any pass yourself</a> · <a href='primes.html'>Watch it sieve the primes</a></p>" },
];

// ---------------------------------------------------------------- the steps' cards

const stepsEl = $("steps");
const cards = STEPS.map((s, i) => {
  const sec = document.createElement("section"); sec.className = "step";
  sec.innerHTML = "<div class='card'><p class='ch-num mono'>" + (i + 1) + " of " + STEPS.length + "</p><h2>" + s.title + "</h2><div class='body'></div></div>";
  stepsEl.appendChild(sec); return sec;
});
function fill() {
  cards.forEach((c, i) => { c.querySelector(".body").innerHTML = STEPS[i].text(view); });
  drawStrip($("strip"), view, false);
  stepsEl.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => follow(b.dataset.go)));
}
let unseenAt = 0, started = false;
function follow(what) {
  if (what === "unseen") { unseenAt = (unseenAt + 1) % INSIDE.unseen.length; show(src.unseen(unseenAt)); }
  else if (!started) { started = true; show(src.start()); }  // the first: two rings into the run, the head in the middle
  else { src.nextHead(); show(src.live()); }
  cards[0].scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
}
function show(v) { view = analyse(v); S.show(view); fill(); step = -1; frame(); }

// ---------------------------------------------------------------- scrolling

// Where the reader is: t = i + f when the reading line is f of the way from step i's middle to step i + 1's. The camera
// rests on a step's view while its card is near the line, and moves between views in between.
let step = -1;
function frame() {
  const narrow = matchMedia("(max-width: 899px)").matches, H = innerHeight;
  const line = narrow ? H * 0.52 + (H * 0.48) * 0.5 : H * 0.5;
  const mids = cards.map((c) => { const r = c.querySelector(".card").getBoundingClientRect(); return r.top + r.height / 2; });
  let t = 0;
  if (line >= mids[mids.length - 1]) t = mids.length - 1;
  else if (line > mids[0]) for (let i = 0; i < mids.length - 1; i++) if (line < mids[i + 1]) { t = i + (line - mids[i]) / (mids[i + 1] - mids[i]); break; }
  const i = Math.floor(t), f = t - i, ease = (x) => x * x * (3 - 2 * x), s = ease(Math.min(1, Math.max(0, (f - 0.2) / 0.6)));
  const a = STEPS[i], b = STEPS[Math.min(i + 1, STEPS.length - 1)];
  S.aim(a.view, b.view, s);
  const now = Math.round(t);
  if (now !== step) {
    step = now;
    S.setChapter(STEPS[now].ch);
    cards.forEach((c, k) => c.classList.toggle("on", k === now));
    revealStrip($("strip"), STEPS[now].ch >= 7);
  }
  $("progress").style.width = (t / (STEPS.length - 1) * 100).toFixed(1) + "%";
}
let pending = false;
const onScroll = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; frame(); }); } };
addEventListener("scroll", onScroll, { passive: true });
addEventListener("resize", onScroll);

themeButton(() => S.retheme());
budgetTable($("budget"));

const w0 = params.get("w");
if (w0 === "live") { started = true; show(src.start()); }
else { unseenAt = w0 && /^unseen[123]$/.test(w0) ? +w0.slice(6) - 1 : 0; show(src.unseen(unseenAt)); }
window.__walk = { view: () => view, step: () => step };
