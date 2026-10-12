# lag0-lagging-window-universality - Trajectory

## 2026-10-09: first closure

Pressure: the course and the films paired the context window with the Turing tape. The keeper proposed instead that
the window is what is being read at any given time. Literature check: Schuurmans, Dai, Zanini (2024) show
generalized autoregressive decoding with a lagging window is a Lag system and universal; Pérez, Barceló, Marinkovic
(JMLR 2021) give the idealized unbounded-context result.

Move: `machines/lag0`. A `.tm0` program compiles into a window-3 rule table over tokens `#`, tape symbols, and head
tokens `state:symbol`. Each decode step rewrites the middle cell of the window from its neighbours; one pass is one
Turing step; the tape grows at either end.

Witnesses:

```text
L1 agree on all 8 fixtures (5 from tm0, 3 new: binary-increment, binary-increment-overflow, move-left-past-start)
L2 the five configurations of A -> 1010 appear in the transcript, one per pass
L3 under a recency window every fixture cycles or stalls
```

Open: weights realization; window of two.

## 2026-10-12: the rule table as transformer weights (`machines/lag1`)

Pressure: the lag0 step was a lookup table. Does a transformer forward pass realize it?

Move: `machines/lag1`. Weights derived from the `.tm0` program: one-hot token and position embeddings; two hard-attention
heads read at the last position (previous-token head copies the middle token, two-back head copies the left); MLP 1 with
one ReLU neuron per row of delta per slot (the rule table as neurons); MLP 2 of Boolean-AND units assembling output
features; unembedding `2<f,u_v> - |u_v|` for an exact, unique argmax. Integer weights, Fraction averages.

Witnesses:

```text
LT-1 forward pass = lag0 table on every reachable window, all 8 fixtures (225..2501 windows), 0 mismatches, no ties
LT-2 decode with the forward pass = tm0 (verdict, steps, tape) on all 8 fixtures
LT-3 same weights, recency window: every fixture cycles or stalls
LT-4 MLP 1 neurons = rows of delta x 3 slots
```

Correction recorded: the first statement of the reachable domain allowed two heads around '#' only when equal. The
decoder's own assertion caught `[q0:_ # qa:1]` in a one-cell ring, where the window sees the same cell from two
generations. The domain was widened to any two heads either side of '#', and LT-1 now covers those windows.

Relation: `minimal-transformer` (NAND realized in one transformer step) supplies combinational completeness; the lagging
read discipline supplies the memory. Open: compile the weights from a universal program, for one fixed network; train
toward the derived weights; a window of two.

## 2026-10-12: trained, not derived (`machines/lag2`)

Pressure: lag1's weights were derived. Does training find a step that runs the machine?

Measurement first: every window met by all 8,176 binary-increment inputs of 4..12 bits already occurs on the 14 inputs of
1..3 bits (137 distinct windows). So train only there.

Move: an ordinary transformer (d=32, one softmax-attention layer, 2 heads, ReLU MLP 64), full-batch Adam on next-token
cross-entropy, five seeds (`fixtures/train_lag2.ref.py`, torch); torch-free witness over the committed weights.

```text
TL-1 pure-Python forward = committed predictions on 784/784 reachable windows
TL-2 fits 137/137 training windows (min margin 11.6)
TL-3 windows of all 4..12-bit inputs: 0 not in training
TL-4 decode with the trained step = tm0 on 8190/8190 inputs of 1..12 bits, 5/5 of 256 bits
TL-5 five seeds: fit at step 100 and long runs correct at step 100 (no grokking gap: the forced regime)
TL-6 control, trained on 1..2 bits (56 needed windows missing): fits, long-run accuracy 0.0 in every seed
TL-7 a previous-token head emerges in every seed (0.79..0.90); no clean two-back head (<= 0.32)
TL-8 agrees with lag1's derived net on 137/137 met windows, 486/784 reachable
```

Reading: length generalization is supplied by the lagging read discipline, which makes every long computation out of
finitely many local steps; training only has to cover those steps. Open: train on a universal program's steps.

## 2026-10-12: one network for every program (`machines/lag3`)

Pressure: lag2's network ran one program. Train on the steps of a universal machine instead.

Move: `derived/utm.py` generates a universal Turing machine as a `.tm0` program (34 states, 23 symbols, 559 rules);
described machines over `0 1 _` are encoded on its tape (one-way tape). Measurement first: steps from uniformly random
machines left 700-900 of the held-out machines' steps unseen, all of them the universal machine carrying a write across
tape patterns; random tapes over 0 1 _ and machines biased to travel right closed it (3,200 machines: 3 unseen). Trained
on 4,000 random machines (14,831 steps), three seeds.

```text
U-1 tm0 runs the universal machine correctly on all 7 held-out machines
U-2 exact lagging decode: 70/70 test runs (23.4M decode steps), 6,489 windows met
U-3 lag1 derived weights = exact rule on 250 sampled test windows
U-4 three seeds fit 14,831/14,831 by step 200; recomputed sample matches
U-5 test windows never in training: 0
U-6 the one trained network runs 70/70: decrement, div3, increment, invert, palindrome, parity, unary-add; inputs to 13 cells
```

Seed 0 formed both derived heads (previous token 0.80, two back 0.68). Open: larger machines; a window of two; a
two-way tape in the universal machine.
