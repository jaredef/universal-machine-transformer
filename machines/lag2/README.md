# lag2

[`lag1`](../lag1/README.md) **derived** transformer weights for the lagging-window decoder. `lag2` **trains** them. An
ordinary small transformer, with no structure from `lag1` built in, is trained only on the steps the decoder takes while
adding one to binary numbers of **1 to 3 bits**. Plugged into the lagging decoder, it then adds one to every number up to
12 bits, and to 256-bit numbers.

| | |
|---|---|
| model | learned token and position embeddings (d = 32), one layer of softmax attention with 2 heads, a ReLU MLP (64), an unembedding; read at the last position of the 3-token window |
| training | full-batch Adam, lr 3e-3, cross-entropy on the next token, 3,000 steps, five seeds |
| data | the 137 distinct windows the decoder meets on the 14 inputs of 1–3 bits |
| reference script | `fixtures/train_lag2.ref.py` (needs torch) → `fixtures/lag2_training.json` |
| witness | `derived/lag2.py`, pure Python, recomputes every forward pass from the committed weights |

```sh
./run.sh
```

| Rung | Result |
|---|---|
| **TL-1 REPRODUCES** | the pure-Python forward pass matches the committed predictions on all 784 reachable windows |
| **TL-2 FITS** | correct on all 137 training windows, minimum logit margin 11.6 |
| **TL-3 FORCED** | every window met by all 8,176 inputs of 4–12 bits is already a training window (0 new) |
| **TL-4 LENGTH-GENERALIZES** | decoding with the trained step equals `tm0` on all 8,190 inputs of 1–12 bits and on 5 inputs of 256 bits |
| **TL-5 SEEDS** | in all five seeds, the data is fit and every long input runs correctly at the **same** checkpoint (step 100): no grokking gap |
| **TL-6 UNFORCED** | trained on 1–2-bit inputs only (56 needed windows missing), every seed fits its data and fails **every** longer input; it guesses 22–42 of the 56 missing steps, and one wrong step derails a run |
| **TL-7 ATTENTION** | measured: in every seed one head puts 0.79–0.90 of its attention on the previous token, as `lag1`'s derived head does; no seed forms a clean two-back head (at most 0.32) |
| **TL-8 DERIVED** | measured: the trained step agrees with `lag1`'s derived network on all 137 windows the decoder meets, and on 486 of 784 reachable windows |

## What it shows

- **Length generalization comes from the read discipline.** A lagging window builds a computation of any length from
  the same finitely many local steps. Once the training data contains all of them, the trained step runs inputs of any
  length. Nothing about long inputs had to be learned.
- **Forced, or not.** This is `minimal-transformer`'s distinction (NO-4b and NO-4c) at the scale of a whole
  computation. When the training data forces every step, generalization arrives as soon as the data is fit. When it does
  not, the model fits what it saw and fails, and partial guessing is not enough, because a run is only as good as its worst
  step.
- **Training found a previous-token head without being told to.** It did not separate a clean two-back head; it reads the
  left neighbour through blended attention. The trained network is a different set of weights from `lag1`'s that computes
  the same function on every step the computation needs.

## Boundary

One program (binary increment), one small architecture, five seeds. The trained network learns this program's steps,
not the construction: on reachable windows no computation meets, it differs from `lag1` (TL-8). A network that runs any
program would be trained on the steps of a universal machine.
