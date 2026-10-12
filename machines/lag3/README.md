# lag3

`lag2` trained a network on the steps of **one** program. `lag3` trains **one network for every program**: it learns
the steps of a **universal Turing machine**, which reads a described machine and its input from the tape and runs it.
Trained only while the universal machine ran *random* small machines on short random tapes, the network then runs
machines it never saw (increment, decrement, invert, parity, divisibility by 3, unary addition, a six-state palindrome
checker) on inputs more than twice as long as any in training.

## The universal machine (`derived/utm.py`)

34 states, 23 symbols, 559 rules, generated in Python and written as an ordinary `.tm0` program. Its tape holds a
described machine, then that machine's tape:

```text
*aR;bR;cLx;|bSA;aLx;bSA;$O111        binary increment, on input 0111
```

Each entry is one state of the described machine: three blocks, for reading `0`, `1`, `_`. A block is the symbol to
write (`a b c`), the move (`L R S`), and the next state, in unary (`x` per state number), `A` to accept, or `H` to
reject; `H;` alone means no rule. `*` marks the current state, `$` starts the described machine's tape, and `O I U` mark
its head. One cycle of the universal machine is one step of the described one. The described machine's tape is one-way
infinite (a left move on cell 0 stays), the standard simplification.

## The experiment

| | |
|---|---|
| training data | the 14,831 distinct lagging-window steps the universal machine takes while running **4,000 random machines** (1–5 states; half biased to travel right) on random tapes of 1–6 cells over `0 1 _` |
| model | learned embeddings (d = 64), one layer of softmax attention with 2 heads, a ReLU MLP (256), an unembedding over the decoder's output tokens; read at the last position of the 3-token window |
| training | full-batch Adam, lr 3e-3, 3,000 steps, three seeds (`fixtures/train_lag3.ref.py`, needs torch) |
| witness | `derived/lag3w.py`, pure Python, recomputes the network from the committed weights (about 2 minutes) |

```sh
./run.sh
```

| Rung | Result |
|---|---|
| **U-1 UNIVERSAL** | `tm0` itself, running the universal machine, reproduces each of the seven held-out machines |
| **U-2 LAGGED** | on the lagging-window decoder with the exact rule, all 70 test runs are correct (23.4 million decode steps) |
| **U-3 DERIVED** | `lag1`'s derived transformer weights equal the exact rule on 250 sampled windows the tests meet |
| **U-4 FITS** | all three seeds fit all 14,831 training steps by step 200 with no errors; recomputed here, the committed network matches on a sample |
| **U-5 COVERAGE** | the tests meet 6,489 different steps; **none** is missing from training |
| **U-6 ONE-NETWORK** | the trained network, as the decoder's step, runs **70 / 70** test runs: decrement 12/12, div3 12/12, increment 12/12, invert 8/8, palindrome 10/10, parity 10/10, unary-add 6/6; inputs up to 13 cells |

## What it shows

- **One fixed network runs any machine.** The programs live on the tape, as data; the network only ever learned the
  universal machine's steps. No held-out machine contributed a single training example.
- **Why it generalizes.** The lagging window makes every computation, of any machine and any length, out of the
  universal machine's finitely many local steps. Random small machines on random short tapes visit all of the steps the
  held-out machines need (U-5), so this is the forced regime of `lag2` and `minimal-transformer`, now across programs as
  well as lengths.
- **The coverage had to be earned.** Uniformly random machines left 700–900 needed steps unseen (the universal machine
  carrying a write across particular patterns of tape); varied tapes and machines biased to travel right closed the gap.
  The measurement is part of the result.
- **Attention.** Seed 0 formed both heads the derived construction uses (80% on the previous token, 68% two back);
  seeds 1 and 2 blend them.

## Boundary

The tested machines are small (up to six states) and their inputs moderate (up to 13 cells); larger machines need only
more `x`s and longer scans, the same local steps, but that is argued from U-5's construction, not tested here at scale.
The universal machine is one particular design, with a one-way tape. The network is exact on every step it was trained
on, not on every step the universal machine could ever take; coverage is measured on these tests, not proved for all
inputs.
