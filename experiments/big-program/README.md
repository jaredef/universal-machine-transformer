# A big program: the sieve of Eratosthenes (milestone 5)

Universal machine version 2 ([`../fast-universal-machine`](../fast-universal-machine)) finds a state by its binary address,
so a program of hundreds of states costs about the square of its size per step instead of the fourth or fifth power.
This is the program that makes use of it: a 250-state Turing machine that sieves the primes up to 1,680, run by the
23,844-parameter network trained on version 2's step table. The network never saw it. It is drawn live, on an Ulam
spiral, by [`web/primes.html`](../../web/primes.html).

## The program (`primes.py`, `web/primes.js`)

Cell *n* of the tape holds 1 while *n* may be prime and 0 once it is crossed out (cells 0 and 1 start at 0). For *k* = 2,
3, ...: if cell *k* is still 1, *k* is prime; the machine marks it (writing `_`), walks right to the end of the tape
crossing out every *k*-th cell, walks back to the mark, restores it, and moves on to *k* + 1.

The countdown to the next multiple is held in the machine's states, one chain of *k* states for each prime stride *k*.
That is what makes it big: sieving up to *N* takes the strides of the primes up to sqrt(*N*). Only primes get a chain:
when *k* is composite its cell is already 0 and the machine moves on, so a chain for it would never run. The sizes used
are *N* = *s*² − 1, so that an *s* × *s* Ulam spiral shows 1 to *N* (*s*² itself is composite, but its smallest factor
*s* would need a stride the program does not have).

`python3 primes.py N` builds the machine, runs it directly, and runs it through universal machine version 2 (as an
ordinary Turing machine), comparing the configuration after every step:

```text
sieve to 1680: 250 states, 680 rules; description 5,752 cells
direct: 39,991 steps, qa; primes found 263, correct: True
version 2: matches every step; 9,861,395,831,423 forward passes on the lagging decoder, 246,590,379 per described step (468s)
```

(The decoder's own count, below, is 451 passes higher, a difference of five parts in a hundred billion. `check.py`
estimates the passes from the universal machine's tape length, and I have not tracked down the difference.)

## On the network

Ten trillion forward passes is about two days of computing in a browser at the full decoder's 50 million passes a second.
So the run uses a **delta decoder** (`DeltaRun` in `web/core.js`). From one ring of the transcript to the next, only the
cells near the universal machine's head change, plus a cell added at an end. A pass reads a window of three tokens. If
that window is the same three tokens, at the same place, as one ring earlier, it gives the same output, because the
network is deterministic. That output was a copy of the middle token, or the token there would have changed. So each ring
is the previous one with the windows near its changes read by the network and the rest copied. The first ring is read in
full. Every count is the full decoder's:

- **Forward passes:** the number of windows the full decoder reads.
- **Distinct steps, never in training, disagreeing with the exact rule:** counted over the same windows.

`web/tests/delta-check.js` checks this against the full decoder (`FastRun`) on every showcase program, every curve, and
40 random machines, on both universal machines and their networks. It compares the whole ring at every described step,
the verdict, the forward passes, and the distinct, unseen, and disagreeing windows. All are the same.

`web/tests/primes-check.js` runs the sieve at every size the page offers. It checks every step, as read off the network's
transcript (the cell written and where the head went), against the program run directly, and checks the primes left on
the tape:

| Numbers | States | Rules | Steps | Universal steps | Forward passes | Windows read by the network | Never in training | Disagreeing | Primes right | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 to 120 | 32 | 78 | 945 | 2,866,589 | 1,957,557,384 | 11,467,035 | 3 (of 12,406) | 0 | yes, 30 | 1 |
| 1 to 528 | 108 | 286 | 8,333 | 94,077,667 | 251,087,637,692 | 376,313,333 | 21 (of 24,552) | 0 | yes, 99 | 11 |
| 1 to 960 | 170 | 458 | 18,993 | 331,122,214 | 1,460,894,912,457 | 1,324,493,264 | 34 (of 31,557) | 0 | yes, 162 | 37 |
| 1 to 1,680 | 250 | 680 | 39,991 | 1,325,997,462 | 9,861,395,831,874 | 5,303,997,281 | 48 (of 34,378) | 0 | yes, 263 | 147 |

The steps never seen in training are a few dozen out of tens of thousands. The network gets every one of them right, so
the program runs correctly from start to finish. In Chrome the full sieve to 1,680 takes about 2½ minutes, the
equivalent of about 65 billion forward passes a second, of which the network actually reads about 35 million.

Universal machine version 1 cannot run this program in practice. Its cost per step grows with about the fourth power of
the number of states (`check.py --scale`: 139 billion forward passes per step for a 256-state program, against version
2's 61 million).

## The page

[`web/primes.html`](../../web/primes.html) draws the tape as an Ulam spiral: 1 in the middle, the numbers winding outward
counterclockwise. After every step it reads the cell written and the head's position off the network's transcript.
Cells crossed out by the current prime are tinted, the current prime and the head are marked, and a number is shown as
proven prime once it cannot be crossed out any more. While the machine sweeps prime *k*, every number below *k*² that is
still lit is prime. On the Ulam spiral those numbers form the central *k* × *k* square, so the proven region grows
outward from the middle. The page also runs the program directly, one step behind, only to check the network's steps.
At the end the primes' diagonals are plain.
