# The smallest training set

`lag3`'s network was trained on the steps the universal machine takes while running 4,000 random machines, and it runs
every demonstration program correctly. How much of that training does it need? This experiment trains the same network,
with the same settings, on the steps of the first *N* of those random machines, for *N* from 10 to 16,000. It then
tests each network on everything the demonstrations run: the seven held-out machines on their 70 inputs, Rules 90, 30,
and 110 for 16 generations, and the busy beavers BB(2) and BB(4).

The test is exact. The decoder is deterministic, so a network runs a program correctly if and only if it predicts every
step on that program's exact run correctly. Each network is scored on all 7,982 steps the twelve programs need, without
running the decoder.

## Results

**The universal machine's step table is finite and small: 14,849 entries.** Random machines produce new steps quickly,
then stop. 4,000 machines give 14,831 steps (exactly `lag3`'s training set). After 7,000 machines no new step appeared
in the next 9,000. Every one of the 7,982 steps the twelve programs need is in the table, including the 8-state automaton
machine on tapes of up to 33 cells, far outside the training distribution (up to 5 states, up to 6 cells).

**The smallest training set where every network runs every program is 2,500 random machines.** That is 14,775 steps,
99.5% of the table. At that point the twelve programs still need 15 steps that were never in training, and all six seeds
tried predict all of them correctly. Covering every needed step takes 3,568 machines, and `lag3` used 4,000.

| Random machines | Training steps | Demo steps never in training | Unseen demo steps predicted right* | Unseen pool steps predicted right | Programs correct, per seed (of 12) | Seeds running all 12 |
|---|---|---|---|---|---|---|
| 10 | 5,225 | 4,116 | 55.7% | 52.4% | 0 / 0 / 0 | 0 of 3 |
| 20 | 7,080 | 3,052 | 66.0% | 60.9% | 0 / 0 / 0 | 0 of 3 |
| 50 | 9,323 | 1,882 | 74.9% | 71.9% | 0 / 0 / 0 | 0 of 3 |
| 100 | 11,095 | 1,189 | 83.0% | 82.8% | 0 / 0 / 1 | 0 of 3 |
| 200 | 12,701 | 678 | 92.6% | 92.5% | 1 / 1 / 1 | 0 of 3 |
| 500 | 13,864 | 280 | 98.2% | 98.3% | 8 / 8 / 7 | 0 of 3 |
| 1,000 | 14,509 | 123 | 97.9% | 98.8% | 8 / 7 / 9 | 0 of 3 |
| 2,000 | 14,728 | 39 | 98.2% | 99.2% | 9 / 12 / 12 | 2 of 3 |
| 2,250 | 14,755 | 24 | 97.0% | 98.0% | 9 / 12 / 12 / 11 / 11 / 11 | 2 of 6 |
| **2,500** | **14,775** | **15** | **100%** | **100%** | **12 / 12 / 12 / 12 / 12 / 12** | **6 of 6** |
| 2,750 | 14,792 | 12 | 100% | 100% | 12 / 12 / 12 | 3 of 3 |
| 3,000 | 14,807 | 3 | 100% | 100% | 12 / 12 / 12 | 3 of 3 |
| 3,568 | 14,828 | 0 | | 100% | 12 / 12 / 12 | 3 of 3 |
| 4,000 (`lag3`) | 14,831 | 0 | | 100% | 12 / 12 / 12 | 3 of 3 |
| 16,000 | 14,849 | 0 | | | 12 / 12 / 12 | 3 of 3 |

\*Summed over the twelve programs and the seeds, so a step two programs share counts twice. "Unseen pool steps" are the
table's steps that first appear after machine *N*: steps never seen, from the training distribution. Every network fits
its own training set exactly (zero errors, by Adam step 200–700), so every failure is a failure to generalize.

What the table shows:

1. **Coverage is sufficient but not necessary.** The network predicts most steps it never saw: about 98% from 500
   machines up, falling to 93% at 200, 83% at 100, and about 55% at 10.
2. **Runs fail because one wrong step ruins a run.** At 500 machines the network is right on 98% of unseen steps. But
   each program meets dozens of them, so roughly a third of the runs still fail. Success needs a low miss rate *and*
   few unseen steps, and only the second keeps falling as training grows.
3. **From 100 machines up, the failures are compositional.** Every unseen step a program needs is made only of tokens
   that appear in training, and asks for an output that appears in training. The network has seen every part and missed
   the combination. Below 100 machines, part of the failure is plainer: some steps contain a token, or call for an
   output, that training never showed at all (1,117 of the 4,116 unseen steps at 10 machines).

## Why the last failures happen

At 2,250 machines, the three seeds whose wrong steps were recorded (3, 4, 5) each miss exactly one step a program
needs, and it is the **same step** for all three: `1 0 go_aS:I`. Each loses one program, decrement. (Seed 0's three
misses were not recorded.) In the universal machine, `go_aS` writes 0 and does not move. On reaching the marked cell (`I`, a
1 under the head) it steps back, so the answer is always `ret:` followed by the window's middle token, whatever the left
token is. Training at 2,250 machines has `1 1 go_aS:I → ret:1` and `0 0 go_aS:I → ret:0`, but no window with this head
whose left token is 1 and whose answer is `ret:0`. Retrained and probed (`probe.py`), the network answers:

```text
1 0 go_aS:I   correct ret:0   never seen    -> ret:1 0.63, ret:0 0.37
_ 0 go_aS:I   correct ret:0   never seen    -> ret:0 0.87
1 1 go_aS:I   correct ret:1   in training   -> ret:1 1.00
1 0 go_bS:I   correct ret:0   in training   -> ret:0 1.00
1 0 go_aS:O   correct ret:0   never seen    -> ret:1 0.69, ret:0 0.31
```

It takes a shortcut: a 1 in the left slot pulls the answer toward `ret:1`. The right rule, "copy the middle token", is
in the training data. The same window under the sibling head `go_bS` (write 1, do not move) is there with the right
answer, and the network gets that one right. It did not transfer, because `go_aS:I` and `go_bS:I` are separate tokens
with separate embeddings. That they share a structure ("go, then stay, then on a marked 1") is visible only in their
names, and the network cannot read names. Its generalization follows the data, not the structure of the machine that
produced it.

## Limits

- One architecture and one set of training settings (`lag3`'s). A different model, or more training, could move the
  threshold.
- Training sets are prefixes of one fixed random sequence of machines, so "smallest" means smallest *N* for this
  sequence. A training set chosen by hand from the table could be smaller.
- Three seeds per size, six at the threshold. The threshold is a measurement, not a guarantee: at 2,250, two seeds of
  six still pass.
- Training ran on the Apple GPU (MPS), which is not bit-for-bit repeatable. Predicted probabilities can vary by a few
  points between runs of the same seed; the probed failure above repeated on each retraining.

## Reproduce

```sh
python3 collect.py              # about 17 minutes, standard library only: writes steps.json (committed)
python train.py                 # needs torch; about 2 minutes per network on an Apple GPU; writes results.json
SEEDS=3,4,5 python train.py 2250 2500
python3 analyze.py              # the tables above
python probe.py 2250 3          # the probe above
```
