# lag0-lagging-window-universality - Seed

## Telos

Make executable the claim that an autoregressive decoder's context window is the Turing head's view, not its tape,
and that universality is decided by the read discipline.

```text
transcript (append-only, unbounded)  = tape
context window (bounded)             = head's view (C3), carrying the control state
lagging window                       -> Lag system -> universal
recency window                       -> finite-state
```

## Origin

Derived in the GCCS corpus (github.com/jaredef/gccs), arc `generative-machine-trajectory`, alongside `tm2` (universal
self-description) and `transformer4` (autoregressive decode). This repository carries the witnesses on their own.

## Target substrate

`machines/lag0`: a compiler from `.tm0` programs to a finite window rule table, and a decoder that appends at the
frontier while reading at a lagging index.

## Positive witness

L1: for every `.tm0` fixture, `lag0` agrees with `tm0` on verdict, step count, and tape, including growth of the
tape at both ends. L2: the transcript contains the configurations, one per pass.

## Negative witness

L3: the same rule table under a recency window never computes (cycles or stalls) on any fixture. Removing the
lagging read discipline removes universality; nothing else changed.

## Verification command

```sh
machines/lag0/run.sh
```

## Closure condition

L1–L3 green. Next pressure: realize the rule table as transformer weights (hard attention over the window), so the
same witness runs on a `transformer4`-style forward pass; and reduce the window to two, after Schuurmans, Dai, and
Zanini (arXiv:2410.03170).
