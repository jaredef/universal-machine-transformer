# The smallest network

`lag3`'s network has 211,270 parameters. How small can a trained network be and still be universal, in the sense this
repository uses: exact on every step of the universal machine's step table, so that it runs every program whose steps
lie in the table (every program tried so far)?

This experiment trains the same architecture (one layer of softmax attention, 2 heads, read at the last of 3 positions;
a ReLU MLP; an unembedding) at a range of widths, with **plain gradient descent and nothing tuned**: `lag3`'s recipe,
full-batch Adam at learning rate 0.003, stopping when every step is right. The training data is the universal
machine's whole step table, the 14,849 steps that 16,000 random machines produce
([../smallest-training-set](../smallest-training-set)); the vocabularies are trimmed to the tokens the table uses (429
in, 434 out). A network passes when it gets all 14,849 steps exactly right.

## Results

| Width | MLP | Parameters | Seeds that learn the whole table | Adam steps to fit | Steps still wrong (seeds that did not) |
|---|---|---|---|---|---|
| 64 | 256 | 105,330 | 3 of 3 | 200 | – |
| 48 | 192 | 69,890 | 3 of 3 | 200–300 | – |
| 32 | 128 | 40,594 | 3 of 3 | 300–900 | – |
| 24 | 96 | 28,250 | 3 of 3 | 400–500 | – |
| 16 | 64 | 17,442 | 3 of 3 | 500–600 | – |
| 12 | 48 | 12,614 | 3 of 3 | 600–700 | – |
| 8 | 32 | 8,170 | 3 of 3 | 900–1,200 | – |
| 6 | 24 | 6,092 | 2 of 3 | 1,700–4,600 | 10 |
| 4 | 64 | 4,542 | 0 of 3 | – | 477, 2,214, 900 |
| 4 | 16 | 4,110 | 0 of 3 | – | 1, 1,611, 167 |
| 2 | 8 | 2,224 | 0 of 1 | – | 4,148 |

Widths 6 and smaller ran for up to 20,000 Adam steps; the larger ones fit long before 8,000. Width 2 was stopped
after one seed.

**8,170 parameters, every seed.** At width 8 (MLP 32), every seed learns the whole table, in 900 to 1,200 Adam steps.
That is 26 times smaller than `lag3`, 2.6 times smaller than [`lag1`](../../machines/lag1)'s hand-derived weights
(21,331 nonzero integers), and about 7,300 times smaller than the nearest trained universal network we know of (Xu,
Yang, Li & McAllester, 2026: 59.5 million). At width 6, **6,092 parameters**, two seeds of three learn it; the third
ends 10 steps short. At width 4 none does, though one seed ends a single step short.

Each network that passes was checked twice more:

- `check.py` recomputes it in pure Python from its saved weights (`fixtures/`) on all 14,849 steps: every one right.
- `web/tests/programs-check.js`, given the network (`NET=`), runs the showcase programs end to end through the universal
  machine on the lagging-window decoder and compares every configuration with the program run directly. The 8,170- and
  6,092-parameter networks run them all, for example Collatz from 871: 1,657 steps, 270 million forward passes, every
  configuration right.

The 8,170-parameter network also trains in the browser, on the [training page](../../web/train.html): about a thousand
steps of gradient descent on a laptop GPU, under half a minute, then all twelve programs run.

## What the numbers say

- **Almost all of a small network is vocabulary.** At width 8 the token embedding is 3,432 parameters and the
  unembedding 3,906: 90% of the 8,170. The transformer itself (positions, attention, MLP) is 832. The table has 429
  input tokens because most of them are the universal machine's head tokens, one per state and symbol (34 x 23); a
  universal machine with fewer states or symbols, or embeddings built from a head token's parts, would shrink it further.
- **The limit here is training, not capacity, at least at width 4.** One width-4 seed ends one step short of the whole
  table and two end far short (167 and 1,611 steps), and a wider MLP at width 4 does not help. A training regime aimed
  at the last few steps (weighting the steps still wrong, a learning-rate schedule) may well take width 4 over; that is
  left for later, deliberately, so that the result above is what plain gradient descent does.

## Limits

- One architecture and one optimizer recipe, three seeds per size.
- "Universal" here has the same meaning, and the same limit, as for `lag3`: exact on all 14,849 steps the universal
  machine has been observed to take, measured, not proved for every step it could take.
- Training used the whole table, so the programs' steps are all in training. What generalization does with fewer
  random machines is the other experiment's subject; at width 64 it needs about 2,500 of them.
- Training ran on the Apple GPU (MPS), which is not bit-for-bit repeatable; the saved weights are what was checked.

## Reproduce

```sh
python train.py                       # the sweep (needs torch); about 15 minutes on an Apple GPU
STEPS=20000 python train.py 6:24 4:16 4:64 2:8
python3 analyze.py                    # the table above
python3 check.py fixtures/net_d8_h32_seed0.json --full-vocab net8.json
NET=net8.json node ../../web/tests/programs-check.js
```
