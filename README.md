# turing-complete-transformer

**An autoregressive transformer whose context window lags behind its output is a universal computer, and an ordinary
trained transformer learns to be one.** Four executable witnesses, each checked exactly, build the claim in steps:

| | Witness | The claim it checks |
|---|---|---|
| [`lag0`](machines/lag0) | a Lag-system decoder | A loop that reads three tokens, appends its output, and moves its window forward by one runs any Turing machine, if the window reads the **old** end of the transcript. The same rule table reading the most recent tokens never computes. |
| [`lag1`](machines/lag1) | derived transformer weights | That rule table is realized exactly by a transformer forward pass: two attention heads read the window, and the Turing machine's rule table becomes a layer of MLP neurons, one per rule. |
| [`lag2`](machines/lag2) | a trained transformer, one program | An ordinary transformer trained on the steps of adding one to numbers of 1–3 bits adds one to every number up to 12 bits, and to 256-bit numbers. |
| [`lag3`](machines/lag3) | a trained transformer, every program | One transformer trained on the steps of a **universal Turing machine** running random machines runs seven machines it never saw (increment, decrement, invert, parity, divisibility by 3, unary addition, a palindrome checker): 70 of 70 runs correct. |

```sh
./run.sh            # every witness, in order; pure Python, standard library only, about four minutes
```

**Watch it run:** open [`web/index.html`](web/index.html) in a browser (no server needed). Pick one of the seven
held-out machines, or write your own, give it an input, and watch `lag3`'s trained network run it: the machine you
described, the universal machine's whole tape, and each forward pass (the window it read, where each attention head
looked, what it predicted). Live checks count any step the network never saw in training and any disagreement with the
universal machine's exact rule. The page runs the committed network itself; `web/build_data.py` regenerates its data
after retraining.

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
2024). The four witnesses make that concrete, then show that training finds it.

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

- D. Schuurmans, H. Dai, and F. Zanini, "Autoregressive Large Language Models are Computationally Universal",
  arXiv:2410.03170 (2024).
- J. Pérez, P. Barceló, and J. Marinkovic, "Attention is Turing-Complete", *Journal of Machine Learning Research* 22 (2021).
- W. Merrill and A. Sabharwal, "The Expressive Power of Transformers with Chain of Thought", ICLR 2024.
- H. Wang, "Tag systems and lag systems", *Mathematische Annalen* 152 (1963).
- [minimal-transformer](https://github.com/jaredef/minimal-transformer): forced and unforced generalization, in a
  transformer step that realizes NAND.

The derivation history, including measurements that changed the design, is in [`docs/trajectory.md`](docs/trajectory.md).
These witnesses were derived in the [GCCS](https://github.com/jaredef/gccs) corpus.
