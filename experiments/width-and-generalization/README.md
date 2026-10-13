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
- **It is not a smooth law.** With three seeds width 16 looked worse than width 64; the denser sweep below shows that
  was noise, and that the effect is a threshold: below about one parameter per training step, generalization is better,
  and above it width makes no difference. The link to compositional embeddings is a hypothesis this experiment is
  consistent with, not one it establishes.
- **With enough data, width stops mattering.** From 1,000 machines on, all three widths run nearly every program.

The usual expectation is that bigger networks generalize better. Here, for a task whose structure is hidden in its
tokens, the smallest network that can learn the task at all generalizes best from little data: compression forces it to
share what tokens have in common, and the shared structure carries over to steps it never saw.

## Width 16, and the shape of the curve

With three seeds, width 16 looked worse than width 64, which would contradict "narrower is better". The obvious
suspect was double descent: error on unseen data peaks where a model's parameter count is close to its number of
training examples, and width 16 (17,442 parameters) sits close to the 13,864 steps of 500 machines. So widths 6 to 64
were swept densely, five seeds each, on 500 machines and on 20 (7,080 steps), with the error peak predicted at about
width 13 for 500 machines and about width 8 for 20 (`curve.py`):


Trained on the first 20 random machines: 7,080 training steps
| Width | Parameters | Parameters per training step | Seeds | Unseen steps wrong (mean) | Range over seeds | Programs correct (mean of 12) | Adam steps to fit (mean) |
|---|---|---|---|---|---|---|---|
| 6 | 6,092 | 0.86 | 5 | 43.27% | 39.67–48.62% | 0.0 | 4,610 |
| 8 | 8,170 | 1.15 | 5 | 40.63% | 38.40–41.61% | 0.0 | 1,250 |
| 10 | 10,344 | 1.46 | 5 | 41.97% | 38.80–46.31% | 0.0 | 950 |
| 12 | 12,614 | 1.78 | 5 | 43.56% | 41.63–44.30% | 0.0 | 640 |
| 14 | 14,980 | 2.12 | 5 | 43.62% | 40.75–45.40% | 0.0 | 460 |
| 16 | 17,442 | 2.46 | 5 | 43.37% | 40.44–45.21% | 0.0 | 710 |
| 20 | 22,654 | 3.20 | 5 | 43.22% | 39.40–47.37% | 0.0 | 440 |
| 24 | 28,250 | 3.99 | 5 | 44.22% | 42.64–46.72% | 0.0 | 330 |
| 32 | 40,594 | 5.73 | 5 | 44.54% | 42.59–47.23% | 0.0 | 360 |
| 48 | 69,890 | 9.87 | 5 | 43.46% | 43.02–44.60% | 0.0 | 260 |
| 64 | 105,330 | 14.88 | 5 | 42.79% | 41.64–44.43% | 0.0 | 170 |

Trained on the first 500 random machines: 13,864 training steps
| Width | Parameters | Parameters per training step | Seeds | Unseen steps wrong (mean) | Range over seeds | Programs correct (mean of 12) | Adam steps to fit (mean) |
|---|---|---|---|---|---|---|---|
| 6 | 6,092 | 0.44 | 5 | 1.46% | 0.30–2.34% | 9.4 | 6,630 |
| 8 | 8,170 | 0.59 | 5 | 1.64% | 0.51–3.15% | 9.6 | 1,380 |
| 10 | 10,344 | 0.75 | 5 | 2.86% | 0.91–5.08% | 7.8 | 880 |
| 12 | 12,614 | 0.91 | 5 | 3.57% | 2.03–5.48% | 6.6 | 680 |
| 14 | 14,980 | 1.08 | 5 | 4.06% | 2.44–5.99% | 7.0 | 580 |
| 16 | 17,442 | 1.26 | 5 | 3.96% | 1.62–5.79% | 8.0 | 690 |
| 20 | 22,654 | 1.63 | 5 | 3.51% | 1.62–5.08% | 7.4 | 430 |
| 24 | 28,250 | 2.04 | 5 | 4.12% | 2.44–5.69% | 6.8 | 450 |
| 32 | 40,594 | 2.93 | 5 | 3.63% | 1.52–4.67% | 8.0 | 420 |
| 48 | 69,890 | 5.04 | 5 | 3.31% | 2.23–4.26% | 8.4 | 1,430 |
| 64 | 105,330 | 7.60 | 5 | 3.74% | 2.94–5.18% | 6.6 | 180 |

- **Width 16 is not special.** With five seeds, widths 12 to 64 are indistinguishable at 500 machines (3.3% to 4.1% of
  unseen steps wrong, with overlapping ranges); the three-seed difference was noise.
- **There is no double descent.** Error rises to a plateau near one parameter per training step and stays there to
  width 64, with no second descent.
- **The effect is a threshold, not a slope.** Below about one parameter per training step (widths 6 and 8 at 500
  machines: 0.44 and 0.59), unseen-step error is less than half the plateau's: 1.5% to 1.6% against 3.3% to 4.1%.
  Above it, width makes no difference.
- **On 20 machines the curve is flat** (40.6% to 44.5% wrong at every width), as predicted, since every width there
  has more than 0.86 parameters per training step. But this does not settle the explanation: with 20 machines, many
  unseen steps contain a token or call for an output that training never showed (the smallest-training-set experiment
  counted 569 of 3,052 such steps among the programs' needs), which no network of any width could get right, so this
  regime is limited by missing information as much as by size.

So the working account is: a network too small to memorize its training set must compress it, and the compression
carries over to steps it never saw; a network that can memorize does, at any width. The step table is finite (14,849
steps), so training sets cannot be made large enough here to test whether the threshold moves to wider networks as the
data grows, which is the account's sharpest prediction.

## Limits

- Three to five seeds per cell; the program counts in particular vary from seed to seed.
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
