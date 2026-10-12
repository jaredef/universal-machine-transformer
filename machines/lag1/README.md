# lag1

`lag1` realizes [`lag0`](../lag0/README.md)'s rule table as **transformer weights**. The same lagging-window decoder now
takes each step with a forward pass: embedding, attention, two MLPs, and an unembedding. It still runs any `.tm0`
Turing program, with the context window as the head's view and the append-only transcript as the tape.

| Layer | What it does | What it is in the Turing machine |
|---|---|---|
| embedding | each token → features (boundary, head, tape symbol, state) + a position one-hot | the alphabet (C1) |
| attention | two heads, hard attention, read at the last position: a **previous-token head** copies the middle token, a **two-back head** copies the left one | the head's view (C3) |
| MLP 1 | one ReLU neuron per row `(q, a)` of δ, per slot; fires when the slot is in state `q` reading `a`, writes that row's move, next state, and symbol | **the rule table (C5), as neurons** |
| MLP 2 | ReLU neurons that are Boolean ANDs, assembling the output token's features | the write and the move |
| unembedding | `logit(v) = 2⟨f, u_v⟩ − |u_v|`: maximal, and unique, exactly when `f` is `v`'s features | the next symbol |

All weights are small integers derived from the `.tm0` program, not trained. Activations are exact (attention averages
use `Fraction`), so every claim is checked exactly.

```sh
./run.sh
```

- **LT-1, forward = table.** On every window the decoder can meet, the forward pass equals `lag0`'s rule table
  (exhaustive: 225–2,501 windows per program, no mismatches, no tied logits). The decoder also asserts that every window it
  actually meets lies in that checked set.
- **LT-2, agreement.** Decoding with the forward pass reaches `tm0`'s verdict, step count, and tape on all eight fixtures.
- **LT-3, the removal test.** The same weights under a recency window never compute.
- **LT-4, the rule table is neurons.** MLP 1 has exactly one neuron per row of δ, per slot.

Look inside one step:

```sh
python3 derived/lag1.py explain ../lag0/fixtures/binary-increment.tm0 '0 1 q1:1'
python3 derived/lag1.py params  ../lag0/fixtures/binary-increment.tm0
```

For binary increment: `d_model=77`, 2 attention heads, 18 rule-table neurons and 31 assembly neurons, under 500 nonzero
integer weights.

## Relation to minimal-transformer

[`minimal-transformer`](https://github.com/jaredef/minimal-transformer) shows a one-layer, tied-embedding transformer step
realizing **NAND**, with its weight found by exhaustive search and reached by gradient descent. NAND is functionally
complete, so a fixed-weight step can realize any finite rule table. `lag1` builds one such step directly: every neuron in
its MLPs is a Boolean AND over features, the rule table of a Turing machine becomes a layer of those neurons, and the
**lagging read discipline** supplies the unbounded memory that a single step lacks. Combinational completeness in the
weights, plus a window that can read the transcript again, gives a universal machine.

## Boundary

- **Derived, not trained.** This is an existence proof by construction. Whether gradient descent finds these weights is a
  separate question, of the kind `minimal-transformer` asks for NAND.
- **One network per program.** The weights are compiled from each `.tm0` program. Compiling them from a *universal*
  program gives one fixed network for all machines; Schuurmans, Dai, and Zanini (arXiv:2410.03170) give a universal Lag
  system with 2,027 rules and a window of two.
- **Hard attention**, as in Pérez, Barceló, and Marinkovic (JMLR 2021). Softmax attention with large enough scores
  approximates it; this witness keeps it exact.
- **Multi-symbol tokens.** When the tape grows, one output token decodes to two transcript symbols, as a real tokenizer's
  tokens often span several characters.
