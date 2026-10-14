# A faster universal machine (milestone 1)

Running a described machine on the network costs, per step of that machine, the universal machine's steps times the
length of the transcript (each universal step is one trip around the ring on the lagging-window decoder). For the
universal machine of `lag3` (version 1), that grows roughly as the fifth power of the described machine's number of
states: 140,000 forward passes per step for an 8-state machine, 2.4 million for 14 states. Profiling shows why: version 1
writes the next state in unary, as k x's, and moves the current-state mark to entry k by walking from the rule to the far
left end of the program and back once per x. That is 69% of all the work at 8 states and 89% at 14.

## Version 2 (`utm2.py`)

Version 2 finds entry k in binary. Each entry's separator carries a tag, the number of trailing zeros of the entry's
index, and the next state is written as the levels of k's 1 bits, highest first. "Go forward 2^l entries" is then one
walk to the next separator tagged at least l: starting at entry 0 and taking the bits from the top, every jump starts at a
multiple of 2^(l+1) and lands exactly 2^l entries on. The machine also remembers whether its mark is left or right of the
rule it is reading, so it never walks back to the start of the program between jumps. The rest of the cycle (find the
head, read the rule, write, move) is version 1's.

It has 82 states, 37 symbols, and 2,137 rules (version 1: 34, 23, 559), and runs described machines of up to 256 states.
The system stays universal, because a small universal Turing machine over 0 1 _ fits in far fewer states and can run
anything; above 256 states, version 1's unary scheme would be the fallback.

## Correct

`check.py` runs both universal machines directly, as ordinary Turing machines, and after every step of the described
machine compares its state, head, and tape, read off the universal machine's tape, with the described machine run
directly:

378 programs (300 random), 353,500 described steps, every configuration compared: all match, both versions   (468s)
| Program | Described steps | Forward passes per step, v1 | v2 | Speedup |
|---|---|---|---|---|
| increment(001000) | 8 | 4,030 | 4,464 | 0.9x |
| decrement(100111110) | 12 | 4,198 | 4,604 | 0.9x |
| invert(00010111100) | 12 | 1,926 | 2,105 | 0.9x |
| parity(010110) | 7 | 4,347 | 4,517 | 1.0x |
| div3(11011010) | 9 | 9,822 | 9,480 | 1.0x |
| unary-add(1111111_11) | 12 | 7,577 | 7,658 | 1.0x |
| palindrome(1010101) | 36 | 102,079 | 39,575 | 2.6x |
| dragon (8 bits) | 2,528 | 140,810 | 56,762 | 2.5x |
| fibonacci (8 bits) | 634 | 2,372,079 | 224,735 | 10.6x |
| collatz(27) | 882 | 132,384 | 81,928 | 1.6x |
| rule 90 | 3,000 | 154,090 | 94,068 | 1.6x |
| rule 30 | 3,000 | 189,846 | 96,049 | 2.0x |
| rule 110 | 3,000 | 168,984 | 94,432 | 1.8x |
| bb2 | 9 | 8,204 | 6,870 | 1.2x |
| bb4 | 118 | 42,262 | 23,182 | 1.8x |

## Faster, increasingly with size

The table above is per program. On small programs version 2 gains little or nothing (0.9x to 2.6x): there, the walks
between the program and the head dominate, and those are unchanged, while version 2's tags make the transcript slightly
longer. The gain is in the next-state search, so it grows with the number of states. `check.py --scale` runs the same
8-state computation (the dragon curve's counting machine) with its states spread at random among more:

| States | v1 passes per step | v2 passes per step | Speedup | v2 correct |
|---|---|---|---|---|
| 8 | 272,138 | 67,308 | 4x | yes |
| 16 | 1,478,114 | 185,441 | 8x | yes |
| 32 | 45,323,577 | 942,252 | 48x | yes |
| 64 | 328,580,831 | 3,125,182 | 105x | yes |
| 128 | 7,294,990,310 | 14,258,122 | 512x | yes |
| 256 | 138,712,614,026 | 60,596,628 | 2,289x | yes |

Version 1 above 32 states is measured over its first few steps only (simulating it further takes too long). Each
doubling of the program multiplies version 1's cost by about 19 (roughly the fourth power of the number of states) and
version 2's by about 4 (roughly the square). At 50 states, version 2 needs about 2 million forward passes per described
step, about 25 steps a second in the browser: a few thousand steps in minutes, where version 1 needs hours.

## What remains

- Milestone 2: collect version 2's step table from random machines (it will be bigger: version 2's states times its
  symbols bound the head tokens at about 3,000 against version 1's 829) and check that it closes, as version 1's did.
- Milestone 3: train the smallest network that learns it. Expect more parameters than 8,170; version 1 stays the record.
- The walks between program and head now dominate small programs; placing the rule nearer the head is the next lever.

    python3 check.py            correctness on 378 programs, both versions, and cost per program (about 8 minutes)
    python3 check.py --scale    cost against the number of states
