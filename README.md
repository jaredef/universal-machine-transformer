# turing-complete-transformer

**An autoregressive transformer whose context window lags behind its output is a universal computer, and an ordinary
trained transformer learns to be one, with as few as 8,170 parameters.** As far as we can find, that is the smallest
*trained* transformer shown to act as a universal computer, about 7,300 times smaller than the nearest comparable
result, and smaller than this repository's own hand-built one (see [How small it is](#how-small-it-is)). Six executable
witnesses, each checked exactly, build the claim in steps:

| | Witness | The claim it checks |
|---|---|---|
| [`lag0`](machines/lag0) | a Lag-system decoder | A loop that reads three tokens, appends its output, and moves its window forward by one runs any Turing machine, if the window reads the **old** end of the transcript. The same rule table reading the most recent tokens never computes. |
| [`lag1`](machines/lag1) | derived transformer weights | That rule table is realized exactly by a transformer forward pass: two attention heads read the window, and the Turing machine's rule table becomes a layer of MLP neurons, one per rule. |
| [`lag2`](machines/lag2) | a trained transformer, one program | An ordinary transformer trained on the steps of adding one to numbers of 1–3 bits adds one to every number up to 12 bits, and to 256-bit numbers. |
| [`lag3`](machines/lag3) | a trained transformer, every program | One transformer trained on the steps of a **universal Turing machine** running random machines runs seven machines it never saw (increment, decrement, invert, parity, divisibility by 3, unary addition, a palindrome checker): 70 of 70 runs correct. |
| [`lag4`](machines/lag4) | the same network, drawing | Not retrained: given an 8-state Turing machine that computes elementary cellular automata, `lag3`'s network draws Rule 90 (the Sierpiński triangle), Rule 30, and Rule 110 correctly, every row, from tens to hundreds of millions of forward passes. |
| [`lag5`](machines/lag5) | the same network, the busy beaver | Not retrained: the network runs the four-state **busy beaver** champion (107 steps, 13 ones, the proven maximum) through every one of its configurations, in order, in 4,986,912 forward passes. |

```sh
./run.sh            # every witness, in order; pure Python, standard library only, about eight minutes
```

**Watch it run:** open [`web/index.html`](web/index.html) in a browser (no server needed). Pick one of the seven
held-out machines, or write your own, give it an input, and watch `lag3`'s trained network run it: the machine you
described, the universal machine's whole tape, and each forward pass (the window it read, where each attention head
looked, what it predicted). Live checks count any step the network never saw in training and any disagreement with the
universal machine's exact rule. The page runs the committed network itself; `web/build_data.py` regenerates its data
after retraining.

**Watch it draw:** open [`web/draw.html`](web/draw.html). The same network runs the universal machine, which runs a
cellular-automaton machine, and the picture appears row by row: the Sierpiński triangle, Rule 30, Rule 110, or any of
the 256 rules, each row checked against the automaton computed directly.

**Watch the busy beaver:** open [`web/beaver.html`](web/beaver.html). The four-state champion, the longest-running
halting machine of its size, runs one step at a time on the same network: its tape, its head, the rule it used, and the
forward passes each step took, checked against the champion run directly.

**Programs it never saw:** open [`web/programs.html`](web/programs.html). The same network runs real programs written
for that page: the Collatz problem (from 27, climbing to 9,232; from 871, to 190,996), a binary counter, a bubble sort,
binary addition, and counting in binary, each checked configuration by configuration against the program run directly.
`node web/tests/programs-check.js` runs them all: every configuration matches, none needs a step outside training.

**Fractal curves:** open [`web/curves.html`](web/curves.html). The 8,170-parameter network runs three more programs it
never saw, Turing machines that count and compute, for every number, which way a line turns: the paperfolding sequence,
which draws the Heighway dragon; the Thue–Morse sequence, which draws the Koch curve (a turn of +60° where it changes and
-120° where it repeats); and the Fibonacci word, computed by counting in the Fibonacci (Zeckendorf) number system, which
draws the Fibonacci word fractal. About 1–2 billion forward passes for a thousand turns, in under a minute;
`node web/tests/curves-check.js` checks every turn on both networks. The programs page also computes the Fibonacci
numbers themselves, in binary.

**Train it yourself:** open [`web/train.html`](web/train.html). The network trains in the page, on your GPU (WebGPU) or
across your processor's cores, on the steps of the first *N* random machines (the slider reproduces the
[smallest-training-set](experiments/smallest-training-set) experiment's training sets). Then it is tested exactly on the
twelve demonstration programs, with three.js views of the whole step table lighting up and of the token embeddings
moving as it learns. A network you train there can replace the committed one on the other pages. The trainer is
hand-written: `web/train-core.js` (plain JavaScript, checked against PyTorch's gradients by `web/tests/gradcheck.py`) and
`web/train-gpu.js` (WebGPU compute shaders, checked against it by `web/tests/gpu-check.html`). On an Apple GPU one
full-batch step takes about 35 ms, as fast as PyTorch on the same GPU, so lag3's 3,000-step recipe takes under two minutes.

## How small it is

**8,170 parameters**, trained by plain gradient descent. The [smallest-network experiment](experiments/smallest-network)
trains `lag3`'s architecture at a range of widths with `lag3`'s recipe unchanged (full-batch Adam, learning rate
0.003) on the universal machine's whole step table. At width 8 every seed learns all 14,849 steps exactly, in about a
thousand Adam steps; at width 6 (**6,092** parameters) two seeds of three do. Each network that passes is recomputed
in pure Python from its saved weights and checked on every step, and runs the showcase programs end to end. The
8,170-parameter network trains in [`web/train.html`](web/train.html) in under half a minute on a laptop GPU.

| Network | Parameters | |
|---|---|---|
| width 8, trained | **8,170** | every seed exact; 90% of it is token embedding and unembedding, the transformer itself is 832 |
| width 6, trained | 6,092 | two seeds of three exact |
| `lag1`, hand-built | 21,331 | nonzero integer weights, none learned |
| `lag3`, trained | 211,270 | the network these pages run and the witnesses check |

`lag3` is the committed reference: one transformer layer, two attention heads, a 64-wide residual stream, a 256-unit
MLP, read over a window of three tokens, 211,270 parameters (0.85 MB at 32-bit precision):

| Part | Parameters |
|---|---|
| Unembedding (1,670 output tokens) | 108,550 |
| Token embedding (829 input tokens) | 53,056 |
| The transformer itself (positions, attention, MLP) | 49,664 |
| **Total** | **211,270** |

Only 429 of its input tokens and 434 of its outputs ever occur; the smaller networks above have only those rows.

**Trained transformers that act as universal computers** (run programs they were not trained on):

| Work | Parameters | What it runs |
|---|---|---|
| **this repository**, width 8 | **8,170** | a universal Turing machine (34 states, 23 symbols, 559 rules); trained on the steps of random machines; runs the held-out machines, cellular automata, the busy beaver, Collatz, sorting, addition |
| this repository, `lag3` | 211,270 | the same, the committed reference |
| Xu, Yang, Li & McAllester (2026) | 59.5 million | MicroPy, a small universal programming language; trained on random programs; runs human-written programs including addition, multiplication, SAT |
| Schuurmans, Dai & Zanini (2024) | frontier scale (Gemini 1.5 Pro; undisclosed) | a universal Lag system with 2,027 rules, by prompting, not training |

Related work that is not the same comparison: Hou et al. (2024, *Turing Programs*) train a 150-million-parameter
transformer for length generalization on specific algorithmic tasks; Pérez et al. (2021), Giannou et al. (2023),
Li & Wang (2025), and Bates (2026) *construct* Turing-complete transformers by hand rather than training them. So does
this repository: [`lag1`](machines/lag1) realizes the same universal machine with **21,331** nonzero integer weights,
none learned, and gradient descent finds one 2.6 times smaller.

What makes the comparison less than exact:

- **The record is for trained networks.** Hand-built constructions elsewhere may be smaller; few report parameter
  counts, and ours (`lag1`) is not.
- **Turing completeness belongs to the system,** not to the weights alone: the network is the step of a loop whose
  three-token window reads the *old* end of an unbounded transcript. That read discipline is not standard decoding, and
  it is what lets so small a network suffice: every computation becomes finitely many local steps.
- **The network's exactness is measured, not proved.** It computes the universal machine's step exactly on all 14,849
  steps the universal machine has been observed to take: no new step appeared in 9,000 further random machines or in
  any of the twelve demonstration programs ([the smallest-training-set experiment](experiments/smallest-training-set)).
  A proof that no other step is reachable is open.
- **The programs differ in kind.** Xu et al.'s interpreter handles procedure calls and recursion, with longer traces;
  ours runs a Turing machine encoding, which puts more of the structure in the universal machine.
- **"As far as we can find"** means the main lines of work (constructions, trained universal executors, prompted
  language models), searched in October 2026. This field moves quickly.

It could be smaller still. Width 4 (4,110 parameters) comes within one step with plain training, so a training regime
aimed at the last few steps may get there. And 90% of the width-8 network is vocabulary, which scales with the
universal machine's states times symbols: a smaller universal machine, or head-token embeddings built from their
parts, would shrink it further.

## The idea

A Turing machine has a tape, a head that sees a bounded part of it, a finite control, and a finite rule table. The
tempting match for an autoregressive transformer pairs the tape with the context window. It is the wrong match:

| Turing machine | Autoregressive transformer |
|---|---|
| the tape | the **transcript**: every token emitted, unbounded, append-only |
| the head's view | the **context window**: a bounded view, read all at once through attention |
| the control state | nothing hidden: state is written into the transcript and read back through the window |
| the rule table | the fixed weights; one forward pass is one step |
| write and move | append one token; the window advances |
| a halting state | a stop token (here, a halt token) |

Whether such a loop can compute anything depends on **where the window reads**. If it holds the most recent tokens, as
in ordinary decoding, the next token depends only on the last few, and the whole loop is a finite-state machine. If it
reads the old end of the transcript while tokens are appended at the new end, the transcript is a queue, and a machine
that reads the front of a queue and writes the back is a **Lag system**, which is universal (Schuurmans, Dai, and Zanini,
2024). The witnesses make that concrete, then show that training finds it.

## How the witnesses work

- **`lag0`** compiles any `.tm0` Turing program into a finite rule table over windows of three tokens. A configuration
  is written into the transcript as a ring (`# c1 c2 ... cn`, with the head as a `state:symbol` token); each step rewrites
  the middle of the window from its two neighbours, so one pass of the window around the ring is one Turing step. It
  agrees with `tm0` (verdict, steps, tape) on every fixture.
- **`lag1`** derives integer transformer weights from the program: a previous-token head and a two-back head (hard
  attention), an MLP with one neuron per rule of δ per slot, a second MLP of Boolean ANDs, and an unembedding whose argmax
  is exact. It equals `lag0`'s table on every window the decoder can meet (exhaustive), using exact arithmetic.
- **`lag2`** trains a small transformer (one attention layer, two heads) with gradient descent. The decoder only ever
  takes 137 different steps on every number up to 12 bits, and all of them occur on numbers of 1–3 bits, so training there
  is enough: the fit and the generalization arrive at the same training step. A control trained on 1–2 bits fits and fails
  every longer input.
- **`lag3`** generates a universal Turing machine (34 states, 23 symbols, 559 rules) as a `.tm0` program. A described
  machine and its input are encoded on its tape. One network is trained on the universal machine's steps while it runs
  4,000 random machines on random tapes of up to 6 cells, then tested on held-out machines with inputs up to 13 cells.
  Every step the tests need occurred in training, and the network runs every test.
- **`lag4`** reuses `lag3`'s network without retraining. An 8-state, 18-rule machine computes any elementary cellular
  automaton (it writes each new cell one place to the right, so every row stays anchored on a one-way tape). On the
  universal machine, the network draws Rules 90, 30, and 110; each run needs only about 3,000 distinct steps, all of which
  occurred in training.
- **`lag5`** runs the two- and four-state busy beaver champions on the same network, with a one-state prelude that gives
  them room on the left of the one-way tape, and checks every configuration of each run, in order.

Each trained network is committed as JSON and recomputed by its witness in pure Python; the training scripts
(`fixtures/*.ref.py`) need `torch` (see `requirements-train.txt`) and use fixed seeds; re-running `lag2`'s was checked to
reproduce its committed weights exactly.

## What is and is not claimed

- **Turing completeness is a property of the style, under its idealization.** The transcript is unbounded; every real
  machine, this one included, is finite. That is the same caveat as for any computer.
- **`lag1` uses hard attention**, as the classic result of Pérez, Barceló, and Marinkovic (2021) does. Softmax attention
  with large scores approximates it; the trained networks in `lag2` and `lag3` use ordinary softmax attention.
- **The window here is three tokens**, so each step is an easy local rewrite. Schuurmans, Dai, and Zanini prove the
  result with a window of two and a universal Lag system of 2,027 rules.
- **The trained networks are exact on the steps they were trained on**, not on every step that could ever occur.
  Generalization in `lag2` and `lag3` is measured, not proved: coverage of the tested runs is checked step by step.
- **`lag3`'s universal machine** has a one-way tape for the described machine (a left move on cell 0 stays), a standard
  simplification, and was tested on machines of up to six states with inputs up to 13 cells.

## Sources

- R. Xu, C. Yang, Y. Li, and D. McAllester, "Training Transformers as a Universal Computer", arXiv:2604.25166 (2026).
- K. Hou, D. Brandfonbrener, S. Kakade, S. Jelassi, and E. Malach, "Universal Length Generalization with Turing
  Programs", arXiv:2407.03310 (2024).
- A. Giannou et al., "Looped Transformers as Programmable Computers", ICML 2023.
- Q. Li and Y. Wang, "Constant Bit-size Transformers Are Turing Complete", NeurIPS 2025.
- J. Bates, "A PyTorch Library of Turing-Complete Neural Networks", arXiv:2605.08150 (2026).
- D. Schuurmans, H. Dai, and F. Zanini, "Autoregressive Large Language Models are Computationally Universal",
  arXiv:2410.03170 (2024).
- J. Pérez, P. Barceló, and J. Marinkovic, "Attention is Turing-Complete", *Journal of Machine Learning Research* 22 (2021).
- W. Merrill and A. Sabharwal, "The Expressive Power of Transformers with Chain of Thought", ICLR 2024.
- H. Wang, "Tag systems and lag systems", *Mathematische Annalen* 152 (1963).
- [minimal-transformer](https://github.com/jaredef/minimal-transformer): forced and unforced generalization, in a
  transformer step that realizes NAND.

## Experiments

- [**The smallest network**](experiments/smallest-network): how small the trained network can be. With `lag3`'s
  recipe unchanged, a network of 8,170 parameters (width 8) learns the universal machine's whole step table exactly,
  every seed; 6,092 (width 6) two seeds of three; at width 4 plain training falls just short.
- [**The smallest training set**](experiments/smallest-training-set): how much training the universal step needs. The
  universal machine's whole step table is finite, 14,849 entries. Trained on the first 2,500 random machines instead of
  4,000, every seed tried still runs all twelve demonstration programs, predicting steps it never saw. Below that, the
  failures are systematic: the network takes a shortcut that the structure of the machine rules out but the data does not.

The derivation history, including measurements that changed the design, is in [`docs/trajectory.md`](docs/trajectory.md).
These witnesses were derived in the [GCCS](https://github.com/jaredef/gccs) corpus.
