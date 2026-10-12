# lag0

`lag0` runs a Turing machine on an autoregressive loop whose context window **lags behind the frontier**: a Lag
system. It is the executable witness for this claim:

```text
the context window is not the tape; it is the head's view.
the tape is the transcript: unbounded and append-only.
whether the loop is universal depends on where the window reads.
```

| Turing machine | lag0 |
|---|---|
| tape | the transcript: every token emitted, never overwritten, never deleted |
| head's view (C3) | the window: three tokens, starting at index `k` |
| control state | a head token `state:symbol` written in the transcript; no hidden register |
| rule table δ | the compiled rule table: a finite map from windows to tokens |
| one step | read the window, append the output, advance `k` by one |
| halting state | no rule matches a window whose middle is an accepting or rejecting head |

`lag0` compiles any `.tm0` program into a rule table. A configuration is written into the transcript as a ring
`# c1 ... cn`; each decode step rewrites the middle cell of the window from its two neighbours, so one pass of the
window around the ring is one Turing step, and the tape grows at either end when the head walks off it.

Run the verification ladder:

```sh
./run.sh
```

- **L1, agreement.** On every `tm0` fixture and on `lag0`'s own (binary increment, overflow to a longer tape, and
  growth to the left), `lag0` reaches the same verdict, step count, and tape as `tm0`.
- **L2, configurations.** The transcript contains the Turing machine's configurations, one per pass.
- **L3, the removal test.** The same rule table under a **recency** window (the most recent three tokens, as in
  ordinary decoding) never computes: every fixture cycles or stalls. Only the read discipline changed.

Watch it:

```sh
python3 derived/lag0.py ../tm0/fixtures/hex-nibble-a.tm0 --passes    # one configuration per pass
python3 derived/lag0.py ../tm0/fixtures/hex-nibble-a.tm0 --trace     # every window and what it appends
python3 derived/lag0.py fixtures/binary-increment.tm0 --rules        # the size of the compiled rule table
python3 derived/lag0.py fixtures/binary-increment.tm0 --discipline recency
```

## Boundary

The window here is three tokens wide, which makes each Turing step a pass of local rewrites. Schuurmans, Dai, and
Zanini (arXiv:2410.03170, 2024) prove the stronger result with a window of two, and show that a universal Turing
machine needs only 2,027 Lag rules. The rule table here is a lookup table, the simplest realization of fixed weights;
realizing it as attention weights is the next pressure, in the style of `transformer4`.
