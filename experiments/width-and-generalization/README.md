# Width and generalization

The [smallest-network experiment](../smallest-network) trained the universal step at widths down to 6 and found, along
the way, that the narrower the network, the more its head-token embeddings are "state + symbol": a token such as
`go_aS:I` is one opaque input, but a squeezed network learns to represent it as its state plus its symbol (R² of that
fit: 0.18 at width 64, 0.28 at 16, 0.45 at 8, 0.51 at 6; a shuffled baseline gives 0.14). Does that learned structure
help it generalize?

This experiment trains widths 8, 16, and 64 on the steps of the first N of `lag3`'s random machines (the training sets
of [the smallest-training-set experiment](../smallest-training-set)), with one recipe for every width: full-batch Adam,
learning rate 0.003, stopping when every training step is right. Each network is then scored on every step it never
saw, and on the twelve demonstration programs (a program runs correctly when every step it needs is right). Three seeds
per cell.

## Results

Programs run correctly (of 12), per seed
| Random machines | width 8 (8,170) | width 16 (17,442) | width 64 (105,330) |
|---|---|---|---|
| 200 | 1 / 1 / 3 | 0 / 1 / 1 | 1 / 1 / 1 |
| 500 | 12 / 10 / 12 | 6 / 8 / 8 | 5 / 5 / 8 |
| 1,000 | 9 / 11 / 12 | 11 / 11 / 10 | 11 / 12 / 10 |
| 2,000 | 12 / 12 / 11 | 12 / 12 / 11 | 12 / 12 / 11 |
| 2,500 | 12 / 12 / 12 | 12 / 12 / 12 | 12 / 12 / 11 |

Steps never seen in training that the network gets right (all seeds pooled)
| Random machines | width 8 | width 16 | width 64 |
|---|---|---|---|
| 200 | 94.9% | 87.8% | 89.7% |
| 500 | 98.4% | 94.9% | 96.4% |
| 1,000 | 99.4% | 98.7% | 98.9% |
| 2,000 | 99.7% | 98.6% | 98.3% |
| 2,500 | 100.0% | 99.5% | 97.7% |

Runs that did not learn their whole training set: 0

## The control: the same training length for every width

Stopping when the training set is learned gives a narrow network more training: width 8 needs about 1,200 Adam steps,
width 64 about 200. So the two training sets where the widths differ most were rerun with every width trained for
exactly 1,500 steps:

Programs run correctly (of 12), per seed
| Random machines | width 8 (8,170) | width 16 (17,442) | width 64 (105,330) |
|---|---|---|---|
| 200 | 1 / 1 / 2 | 1 / 1 / 1 | 1 / 1 / 1 |
| 500 | 11 / 10 / 9 | 6 / 8 / 9 | 5 / 8 / 8 |

Steps never seen in training that the network gets right (all seeds pooled)
| Random machines | width 8 | width 16 | width 64 |
|---|---|---|---|
| 200 | 95.2% | 90.5% | 92.2% |
| 500 | 98.6% | 96.4% | 97.5% |

Runs that did not learn their whole training set: 0

## What it shows

- **The narrowest network generalizes best.** At every training size, width 8 gets more of the steps it never saw right
  than width 16 or 64, and the difference is largest where data is scarce: from 500 random machines, width 8 runs 10 to
  12 of the twelve programs (stopping at fit) where width 64 runs 5 to 8. The 64-wide network needed about 2,500 machines
  for every seed to run every program in the earlier experiment.
- **It is not only training length.** With every width trained for the same 1,500 steps, width 64 improves a little
  (92.2% of unseen steps right from 200 machines, against 89.7% when stopping at fit), but width 8 still leads at both
  sizes (95.2%, and 98.6% against 97.5% at 500 machines).
- **It is not monotonic in width.** Width 16 does worse than width 64 in both runs, although its embeddings are more
  compositional. So "narrower generalizes better" holds here for the narrowest network, not as a smooth law, and the
  link to compositional embeddings is a hypothesis this experiment is consistent with, not one it establishes.
- **With enough data, width stops mattering.** From 1,000 machines on, all three widths run nearly every program.

The usual expectation is that bigger networks generalize better. Here, for a task whose structure is hidden in its
tokens, the smallest network that can learn the task at all generalizes best from little data: compression forces it to
share what tokens have in common, and the shared structure carries over to steps it never saw.

## Limits

- Three seeds per cell; the program counts in particular vary from seed to seed.
- One architecture, one optimizer recipe, and one task. The recipe here (vocabularies trimmed to the tokens the table
  uses, stopping at fit) is not the earlier experiment's (full vocabularies, a fixed 3,000 steps), and the numbers are
  only comparable within this experiment: this recipe alone lets width 64 run more programs from 1,000 machines than the
  earlier one did.
- R² measures only additive structure in the embeddings; what width 8 shares beyond that is not measured.

## Reproduce

```sh
python train.py                                        # the grid (needs torch); about 15 minutes on an Apple GPU
FIXED=1500 python train.py 8:200 16:200 64:200 8:500 16:500 64:500
python3 analyze.py; python3 analyze.py results_fixed1500.json
```
