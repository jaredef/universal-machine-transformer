// inside.html: the explorer. One forward pass in 3D (inside-scene.js), stage by stage, with buttons to step the window.
import { INSIDE, vin, vout, comma, f3, pct, POS, fmtP, code, describe, maxAbs, windows, analyse, drawStrip, revealStrip,
         budgetTable, CH, createScene, themeButton } from "./inside-scene.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const src = windows();
let unseenAt = -1, view = null;

const stage = $("stage");
stage.querySelector(".v-msg").remove();
const S = createScene(stage, $("tip"), { controls: true });
let cur = Math.max(0, Math.min(CH.length - 1, +params.get("ch") || 0));

function show(v) {
  view = analyse(v);
  drawStrip($("strip"), v, cur >= 7);
  $("w-pass").textContent = comma(v.pass);
  $("w-pass-label").textContent = v.live ? "Forward pass" : "Forward pass, of " + comma(INSIDE.passes);
  $("w-note").textContent = v.live ? "From the sieve to 120, running live in this page." :
    "Never-seen window " + (v.unseen + 1) + " of 3, from the end of the sieve to 120.";
  const at = vin[v.ids[1]] === "#" ? "the ring's start" : "cell " + comma(v.at) + " of " + comma(v.ringLen - 1);
  $("w-sub").textContent = "The transcript around the three tokens the network is reading, at " + at +
    ". It writes the token for the middle one's place in the next ring.";
  S.show(v);
  chapter(cur, true);
}

function chapter(c, sameView) {
  const changed = c !== cur; cur = c;
  document.querySelectorAll("#chapters button").forEach((b, i) => { b.classList.toggle("on", i === c); b.setAttribute("aria-current", i === c ? "step" : "false"); });
  $("ch-num").textContent = (c + 1) + " of " + CH.length;
  $("ch-title").textContent = CH[c].title;
  $("ch-text").innerHTML = TEXT[c](view);
  $("prev").disabled = c === 0; $("next").disabled = c === CH.length - 1;
  revealStrip($("strip"), c >= 7);
  S.setChapter(c, { fly: "always" });
  if (changed || !sameView) S.goTo(CH[c].view);
}

const TEXT = [
  (v) => "<p>The network reads three tokens of the transcript:</p><ul class='toks'>" + [0, 1, 2].map((p) =>
    "<li><b>" + POS[p] + "</b> " + code(vin[v.ids[p]]) + ", " + describe(vin[v.ids[p]]) + "</li>").join("") +
    "</ul><p>Its whole job is the token that goes at <b>c</b>'s place in the next ring. Below, from the top left: the " +
    "embedding table and attention, then the MLP, the unembedding, and the probabilities, with the numbers this pass computes.</p>" +
    "<p class='muted'>" + (v.trained ? "This window was in its training data." : "It never saw this window in training.") + "</p>",
  (v) => "<p>Each token has a row of 8 numbers in the embedding table <b>E</b>: 1,351 tokens, 10,808 parameters, " +
    "45% of the network. The three rows are copied out (highlighted in the table).</p><p>Then a position vector from <b>P</b> " +
    "is added to each, so the same token reads differently as l, c, or r. These three vectors of 8 numbers are all the " +
    "network knows about the window.</p>",
  (v) => "<p>Three 8 × 8 matrices. Only r's position asks a question, its <b>query</b> q = Wq·x<sub>r</sub>, the way a " +
    "language model predicts from its last token. Every position answers with a <b>key</b>, k = Wk·x, and offers a " +
    "<b>value</b>, v = Wv·x.</p><p>The 8 numbers are split between two attention heads of 4 each, " +
    "<span class='hk h1'>head 1</span> and <span class='hk h2'>head 2</span>.</p>",
  (v) => {
    const a = v.f.attention, line = (h) => "<span class='hk h" + (h + 1) + "'>Head " + (h + 1) + "</span> puts " +
      [0, 1, 2].map((p) => "<b>" + pct(a[h][p]) + "</b> on " + POS[p]).join(", ") + ".";
    return "<p>Each head compares the query with the three keys (a dot product, divided by 2) and turns the scores into weights " +
      "that add up to 1. The arcs are as thick as the weights.</p><p>" + line(0) + "<br>" + line(1) + "</p><p>Each head then takes " +
      "that mix of the values; together they make <b>y</b>.</p>";
  },
  (v) => "<p><b>Wo</b> turns the mix into an update, <b>o</b>, which is added to r's own vector: h = x<sub>r</sub> + o. " +
    "That running sum is the <b>residual stream</b>, everything the pass knows so far.</p>" +
    (maxAbs(v.f.o) > maxAbs(v.f.x[2]) ? "<p>The update is larger than what it is added to (its largest number is " + f3(maxAbs(v.f.o)) +
    ", against " + f3(maxAbs(v.f.x[2])) + "): attention has rewritten r's vector with what it found at l and c.</p>" :
    "<p>The update's largest number is " + f3(maxAbs(v.f.o)) + ", against " + f3(maxAbs(v.f.x[2])) + " in r's own vector.</p>"),
  (v) => "<p>32 units. Each takes a weighted sum of the stream (its row of <b>W1</b>) plus a bias, and passes it on only if " +
    "it is positive (ReLU). For this window <b>" + Array.from(v.f.hidden).filter((x) => x > 0).length + " of 32</b> are on.</p><p><b>W2</b> adds what they found back into " +
    "the stream; its largest number goes from " + f3(maxAbs(v.f.h1)) + " to " + f3(maxAbs(v.f.h2)) + ".</p><p class='muted'>Between the two tables, attention and the MLP are 832 parameters.</p>",
  (v) => "<p><b>U</b> has a row for every output the network can write: 1,351 single tokens, 4 pairs (written when the " +
    "machine's head moves past the end of the ring), and writing nothing (the machine halts). An output's score is its " +
    "row's dot product with the stream, plus a bias, and a softmax turns the scores into probabilities. The bars are the " +
    "probabilities.</p><p>The highest:</p><ol class='toks'>" + v.order.slice(0, 3).map((i) =>
    "<li>" + code(vout[i]) + " <span class='muted'>score " + f3(v.f.logits[i]) + ",</span> probability " + fmtP(v.prob(i)) + "</li>").join("") +
    "</ol>",
  (v) => {
    const right = v.wrote === v.exact, w = v.wrote ? code(v.wrote) : "nothing (it halts)";
    let s = "<p>It writes " + w + ". The universal machine's exact rule says " + (v.exact ? code(v.exact) : "halt") + ": " +
      (right ? "<b class='ok'>the same ✓</b>" : "<b class='bad'>different ✗</b>") + ".</p>";
    if (v.live) s += "<p>" + (v.trained ? "This window was in its training data. " : "It never saw this window in training. ") +
      "Then the window moves one token to the right, and the next pass begins. Running the whole sieve to 120 takes " +
      comma(INSIDE.passes) + " of these.</p>";
    else s += "<p>It never saw this window in training, and gives the right answer probability " + fmtP(v.prob(v.f.best)) +
      ". The sieve to 120 reads 12,406 different windows; only 3 were not in training, and all three come in the last 0.2% of its " +
      comma(INSIDE.passes) + " passes, here, as the universal machine jumps to the next state by its binary address. It gets all three right.</p>";
    return s;
  },
];

// ---------------------------------------------------------------- controls

const rail = $("chapters");
CH.forEach((c, i) => {
  const li = document.createElement("li"), b = document.createElement("button");
  b.type = "button"; b.textContent = i + 1; b.title = c.title; b.setAttribute("aria-label", (i + 1) + ": " + c.title);
  b.addEventListener("click", () => { stopTour(); chapter(i); });
  li.appendChild(b); rail.appendChild(li);
});
$("prev").addEventListener("click", () => { stopTour(); if (cur > 0) chapter(cur - 1); });
$("next").addEventListener("click", () => { stopTour(); if (cur < CH.length - 1) chapter(cur + 1); });
// From a window it never saw, the window buttons go back to the live run where it was.
$("next-pass").addEventListener("click", () => { stopTour(); if (view.live) src.step(); unseenAt = -1; show(src.live()); });
$("next-head").addEventListener("click", () => {
  stopTour(); if (view.live || !src.run.window().some(src.isHead)) src.nextHead(); unseenAt = -1; show(src.live());
});
$("unseen").addEventListener("click", () => { stopTour(); unseenAt = (unseenAt + 1) % INSIDE.unseen.length; show(src.unseen(unseenAt)); });

let tour = null;
function stopTour() { if (tour) { clearInterval(tour); tour = null; $("tour").textContent = "▶ Walk through"; } }
$("tour").addEventListener("click", () => {
  if (tour) return stopTour();
  $("tour").textContent = "❚❚ Pause";
  if (cur === CH.length - 1) chapter(0);
  tour = setInterval(() => {
    if (cur < CH.length - 1) chapter(cur + 1);
    else { if (view.live) src.nextHead(); show(view.live ? src.live() : src.unseen(unseenAt)); chapter(0); }
  }, 5200);
});
themeButton(() => S.retheme());
budgetTable($("budget"));

const w0 = params.get("w");
if (w0 && /^unseen[123]$/.test(w0)) { unseenAt = +w0.slice(6) - 1; show(src.unseen(unseenAt)); }
else show(src.start());
chapter(cur);
window.__inside = { view: () => view, chapter: (c) => chapter(c) };
