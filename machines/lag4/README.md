# lag4

`lag3`'s trained network ran seven small machines it never saw. `lag4` gives it something to **draw**: an elementary
cellular automaton, Rule 90 (the Sierpiński triangle), Rule 30 (chaos), or Rule 110 (itself universal). Nothing is
retrained. The automaton is computed by an ordinary Turing machine, that machine is written on the universal machine's
tape as data, and `lag3`'s network runs the universal machine on the lagging-window decoder. Each time the described
machine finishes a generation, its row is read off the tape.

## The automaton machine (`derived/ca.py`)

8 states, 18 rules, for any of the 256 elementary rules. Row *g* has width 2*g*+1 and starts at cell 1 of a one-way
tape. One left-to-right sweep computes the next row, remembering the old values of the two cells behind the head (the
state `cab`), and writes each new cell **one position to the right** of where it belongs. Because each row also widens
by one cell on each side, the shifted row still starts at cell 1: no copying, no second pass, and the tape never needs
to grow leftward. Two end states (`e0`, `e1`) write the last two cells, `back` returns to cell 0, and the start state `g`
marks a finished generation.

```text
g _ -> c00 _ R        cab x -> cbx f(a,b,x) R        cab _ -> eb f(a,b,0) R
eb _ -> back f(b,0,0) L        back 0/1 -> back L        back _ -> g _ S
```

## The witness (`derived/lag4.py`)

```sh
./run.sh
```

| Rung | Result |
|---|---|
| **D-1 MACHINE** | run directly, the automaton machine draws Rules 90, 30, and 110 for 16 generations, equal to the automaton computed directly |
| **D-2 EXACT** | the universal machine on the lagging-window decoder, with its exact rule, draws the same 16 rows for each |
| **D-3 COVERAGE** | every step those runs take occurs in `lag3`'s training data: 3,059 to 3,353 distinct steps per rule, 0 unseen (a measurement) |
| **D-4 NETWORK** | `lag3`'s trained network, recomputed in pure Python from the committed weights, draws all 16 rows of each: 66 to 79 million forward passes per picture |

The network was trained only on random machines of up to five states, on random tapes of up to six cells. The
automaton machine has eight states and its tape reaches 33 cells at 16 generations (97 at 48), and the network never
saw it. It runs it anyway because every individual step the universal machine takes here is one it did see: the
universal machine reduces any described machine to the same few thousand local rewrites. That is measured (D-3), not
proved, and the browser page measures it again on every run.

## In the browser

[`web/draw.html`](../../web/draw.html) runs the same network in JavaScript (about 50 million forward passes a second,
since each of the few thousand distinct windows is evaluated once and then reused) and draws up to 48 rows, checking
each row against the automaton computed directly and counting any step never seen in training and any disagreement
with the exact rule. Any of the 256 rules can be entered; a rule whose steps fall outside training will say so.
`draw.html?rule=30&rows=48&play` starts a run from a link.

| Rule, rows | Forward passes | Distinct steps | Unseen | Disagreements | Rows correct |
|---|---|---|---|---|---|
| 90, 32 | 302,471,332 | 3,059 | 0 | 0 | 32 / 32 |
| 30, 32 | 368,114,608 | 3,353 | 0 | 0 | 32 / 32 |
| 110, 48 | 856,778,722 | 3,290 | 0 | 0 | 48 / 48 |
