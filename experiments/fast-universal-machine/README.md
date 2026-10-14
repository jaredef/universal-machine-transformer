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

## Milestone 2: version 2's step table

`windows.py` collects the lagging-window steps (windows) a universal machine takes from a direct run instead of the
decoder: on the decoder, the transcript is the universal machine's configurations laid end to end, so its windows are the
consecutive triples of that sequence, and only the cells around the head and the joins between rings change from one
configuration to the next. Checked against the actual decoder on 40 runs (20 random machines, each under both versions):
the same windows exactly. That makes collecting cost one universal step per step instead of one forward pass per token,
which is what makes 256-state machines affordable.

`collect2.py` then runs 4,000 random machines of 1 to 256 states (half up to 8, the rest spread on a log scale; random
tapes of 1 to 8 cells; at most 300,000 universal steps each):

```text
1 machines: 2,665 distinct steps (0s)
10 machines: 29,851 distinct steps (0s)
30 machines: 40,711 distinct steps (4s)
100 machines: 81,187 distinct steps (24s)
250 machines: 89,761 distinct steps (57s)
300 machines: 90,448 distinct steps (66s)
500 machines: 94,178 distinct steps (112s)
750 machines: 95,882 distinct steps (170s)
1000 machines: 96,577 distinct steps (242s)
1250 machines: 97,064 distinct steps (301s)
1500 machines: 97,386 distinct steps (367s)
1750 machines: 97,835 distinct steps (431s)
2000 machines: 98,215 distinct steps (494s)
2250 machines: 98,400 distinct steps (546s)
2500 machines: 98,624 distinct steps (608s)
2750 machines: 98,818 distinct steps (675s)
3000 machines: 98,935 distinct steps (733s)
3250 machines: 99,117 distinct steps (800s)
3500 machines: 99,247 distinct steps (867s)
3750 machines: 99,321 distinct steps (924s)
4000 machines: 99,559 distinct steps (994s)
```

The table is about 99,600 steps, against version 1's 14,849, with 1,351 input tokens and 1,356 outputs (version 1: 429
and 434). It is nearly closed but not quite: the last 1,000 machines still added about 650 steps.

What the demonstration programs need, against what the random machines produced:

```text
increment: needs 1,888 steps; 0 never produced by the random machines
decrement: needs 1,849 steps; 0 never produced by the random machines
invert: needs 1,144 steps; 0 never produced by the random machines
parity: needs 1,877 steps; 0 never produced by the random machines
div3: needs 2,492 steps; 0 never produced by the random machines
unary-add: needs 1,318 steps; 0 never produced by the random machines
palindrome: needs 5,788 steps; 0 never produced by the random machines
dragon: needs 5,512 steps; 0 never produced by the random machines
fibonacci: needs 9,014 steps; 0 never produced by the random machines
collatz: needs 5,443 steps; 0 never produced by the random machines
rule 90: needs 4,829 steps; 0 never produced by the random machines
rule 30: needs 5,486 steps; 0 never produced by the random machines
rule 110: needs 5,351 steps; 0 never produced by the random machines
bb2: needs 1,954 steps; 0 never produced by the random machines
bb4: needs 3,804 steps; 0 never produced by the random machines
dragon among 64 states: needs 12,220 steps; 3 never produced by the random machines
dragon among 256 states: needs 18,889 steps; 15 never produced by the random machines
```

Every program of up to 14 states is covered. The dragon curve's machine spread among 64 and 256 states needs 3 and 15
steps the random machines never produced, and they are all of one kind: the machine scanning across a rule during a jump,
over a rare combination of used and unused levels in a next-state field (for example `l K F`, a marked block followed by
levels 5 and 2, both used). Each one's output is "copy the symbol and keep scanning". They occur only in programs with
hundreds of states and particular bit patterns in their next-state numbers, which random sampling reaches rarely. Whether
a trained network gets them right anyway is the first thing milestone 3 measures.

With these vocabularies, the width-8 architecture would have about 24,000 parameters (most of it the token tables).

## Milestone 3: networks for version 2

`train2.py` trains the same architecture and recipe as the smallest-network experiment (full-batch Adam, learning rate
0.003, stopping when every training step is right) on all 99,559 steps the random machines produced, one seed per width:

| Width | Parameters | Learned every training step at | Tail steps right (of 18) | Programs correct (of 17) |
|---|---|---|---|---|
| 64 | 224,268 | Adam step 250 | 17 | 16 |
| 32 | 100,524 | Adam step 750 | 17 | 16 |
| 16 | 47,868 | Adam step 750 | 17 | 16 |
| 8 | 23,844 | Adam step 4,000 | 17 | 16 |

Every network learns the whole table, and every one runs every demonstration program and the dragon curve's machine
spread among 64 states. Each misses one of the 18 tail steps (the steps no random machine produced), and so the machine
spread among 256 states. Each width misses a different one, all of the same kind: a jump at level 3 crossing a rule
whose next-state field holds levels 7, 5, and 4, already used (`N K J`). The right output keeps the jump state; the
networks name the right kind of state with the wrong level (width 8: `jmp2R`, width 32: `jmp4R`). It is the limitation the
smallest-training-set experiment found in version 1: the network sees `jmp3R` and `jmp2R` as unrelated tokens, so with no
example of this context it does not reliably carry the level across it.

The width-8 network (23,844 parameters) is about three times the size of version 1's smallest (8,170), almost all of it
the larger token tables.

## Closing the tail

The steps random machines miss come from a jump crossing the rule being read, whose next-state field is partly used: they
depend on the current state's number, the target's bit pattern, and the jump's level together, so covering them takes many
different transitions executed in programs of hundreds of states. `collect_tail.py` adds 200 "tour" machines of 64 to 256
states, in which every state sends the machine to a fresh random state, so each run executes hundreds of different
transitions. They add about 1,100 steps (100,697 in all; the last of them added none), and coverage was then checked on
three programs the tours were not chosen for:

```text
  increment: needs 1,888 steps; 0 never produced by random machines or tours
  decrement: needs 1,849 steps; 0 never produced by random machines or tours
  invert: needs 1,144 steps; 0 never produced by random machines or tours
  parity: needs 1,877 steps; 0 never produced by random machines or tours
  div3: needs 2,492 steps; 0 never produced by random machines or tours
  unary-add: needs 1,318 steps; 0 never produced by random machines or tours
  palindrome: needs 5,788 steps; 0 never produced by random machines or tours
  dragon: needs 5,512 steps; 0 never produced by random machines or tours
  fibonacci: needs 9,014 steps; 0 never produced by random machines or tours
  collatz: needs 5,443 steps; 0 never produced by random machines or tours
  rule 90: needs 4,829 steps; 0 never produced by random machines or tours
  rule 30: needs 5,486 steps; 0 never produced by random machines or tours
  rule 110: needs 5,351 steps; 0 never produced by random machines or tours
  bb2: needs 1,954 steps; 0 never produced by random machines or tours
  bb4: needs 3,804 steps; 0 never produced by random machines or tours
  dragon among 64 states: needs 12,220 steps; 3 never produced by random machines or tours
  dragon among 256 states: needs 18,889 steps; 0 never produced by random machines or tours
  held out: dragon among 128 states: needs 15,468 steps; 0 never produced by random machines or tours
  held out: collatz(27) among 200 states: needs 16,841 steps; 0 never produced by random machines or tours
  held out: rule 110 among 256 states: needs 21,510 steps; 0 never produced by random machines or tours
```

Every program is covered except the dragon curve's machine among 64 states, which still needs 3 steps no random machine
or tour produced. Retrained on the 100,697 steps (`DATA=steps3.json python train2.py 16 8`):

| Width | Parameters | Learned every training step at | The 3 uncovered steps | Programs correct |
|---|---|---|---|---|
| 16 | 47,868 | Adam step 1,000 | 3 right | **20 of 20** |
| 8 | 23,844 | Adam step 3,250 | 3 right | **20 of 20** |

Both run every program: the fifteen demonstrations, the dragon curve's machine among 64 and 256 states, and the three
held-out ones (the dragon among 128 states, Collatz from 27 among 200, Rule 110 among 256), the last of these needing
21,510 different steps. The 3 steps never in training are right too.

## Milestone 4: in the browser

`web/core.js` now carries both universal machines: version 2 is a port of `utm2.py` (all 2,137 rules and the encoding
checked identical to the Python), and an engine runs whichever machine its data names. `web/build_data2.py` writes
`web/data2.js`: version 2's vocabularies, the 23,844-parameter network, and its 100,697 training windows (1.4 MB).
`web/tests/utm2-check.js` runs every showcase program and curve through both universal machines, each on its own trained
network, on the lagging-window decoder, and checks every step against the program run directly:

| Program | Steps | Version 1 passes per step | Version 2 passes per step | Speedup | Both correct |
|---|---|---|---|---|---|
| Collatz | 882 | 132,385 | 81,928 | 1.6x | yes |
| Binary counter | 1,014 | 13,058 | 9,417 | 1.4x | yes |
| Sort | 648 | 40,890 | 26,534 | 1.5x | yes |
| Add | 1,999 | 190,643 | 58,231 | 3.3x | yes |
| Count in binary | 3,478 | 120,557 | 60,472 | 2.0x | yes |
| Dragon curve | 2,528 | 140,810 | 56,762 | 2.5x | yes |
| Koch curve | 1,014 | 300,575 | 85,086 | 3.5x | yes |
| Fibonacci word fractal | 492 | 221,078 | 64,868 | 3.4x | yes |

dragon curve among 256 states, version 2: 63 steps, 7 turns all right, 45,181,316 passes per step, 0 never in training, 0 disagreeing, 77.0s

On the pages, the curves page offers the 23,844-parameter network running universal machine 2 as a third network, and the
programs page a choice of universal machine. Run there, the Koch curve's 1,023 turns take 352 million forward passes and
6 seconds instead of 1.24 billion, and 1000 + 99 takes 116 million instead of 381 million. A program of 256 states, which
would take universal machine 1 about 139 billion forward passes per step, takes 45 million.

## What remains

- Milestone 5 is done: [`../big-program`](../big-program), a 250-state sieve of Eratosthenes run by the network to 1,680,
  drawn live on an Ulam spiral. Next could be a small Game of Life.
- The walks between program and head now dominate small programs; placing the rule nearer the head is the next lever.

    python3 check.py            correctness on 378 programs, both versions, and cost per program (about 8 minutes)
    python3 check.py --scale    cost against the number of states
