# lag5

A **busy beaver** is the halting machine of a given size that runs longest from a blank tape (Radó, 1962). The values are
found one size at a time, by ruling out every other machine: for two states and two symbols, 6 steps leaving 4 ones; for
four, **107 steps leaving 13 ones** (Brady, 1983); for five, 47,176,870 steps, settled only in 2024. `lag5` runs the
two- and four-state champions on `lag3`'s trained network, which never saw them, and checks the run configuration by
configuration.

## The machines (`derived/bb.py`)

The champions use a two-way tape; the universal machine's described tape is one-way (a left move on cell 0 stays). So a
one-state prelude `w` walks right over a strip of `pad` 0s, erasing them, and hands over to the champion's start state
`A` with exactly as much room on its left as the champion ever uses (10 cells for BB(4), measured by running it).
Halting is accepting.

```text
w 0 -> w _ R    w _ -> A _ S                              the prelude
A _ -> B 1 R    A 1 -> B 1 L    B _ -> A 1 L    B 1 -> C _ L
C _ -> qa 1 R   C 1 -> D 1 L    D _ -> D 1 R    D 1 -> A _ R      BB(4)
```

## The witness (`derived/lag5.py`)

```sh
./run.sh            # about half a minute
```

| Rung | Result |
|---|---|
| **E-1 MACHINE** | on the one-way tape, the prelude and the champion halt after 6 and 107 champion steps, with 4 and 13 ones, and the same final tape as the two-way run |
| **E-2 EXACT** | the universal machine on the lagging-window decoder, with its exact rule, passes through every configuration of the run (state, head, and tape, read off its own tape after each step), in order: 10/10 and 119/119 |
| **E-3 COVERAGE** | every step those runs take occurs in `lag3`'s training data: 1,771 and 2,760 distinct steps, 0 unseen (a measurement) |
| **E-4 NETWORK** | `lag3`'s trained network, recomputed in pure Python, does the same: 119 of 119 configurations of BB(4), 4,986,912 forward passes, halting with 13 ones |

## In the browser

[`web/beaver.html`](../../web/beaver.html) runs the same network in JavaScript and paces it one machine step at a time:
the live tape with the head and its state, the rule just used, the number of forward passes each step took, a
space-time picture of the whole run, and checks against the champion run directly. `beaver.html?bb=2` starts on the
two-state champion; `&speed=0&play` runs it at once.
