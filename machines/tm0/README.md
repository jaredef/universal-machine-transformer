# tm0

The reference Turing machine: a deterministic single-tape transition-table executor. Every other machine in this
repository is checked against it.

```sh
./run.sh                                              # its own verification ladder
python3 tm0.py fixtures/hex-nibble-a.tm0 --trace      # A -> 1010, step by step
```

A `.tm0` program is a header (`blank`, `start`, `accept`, `reject`, `tape`), a `---` line, and one rule per line:
`STATE READ -> NEXT WRITE MOVE`, with `MOVE` one of `L R S`. A missing rule rejects. From the GCCS corpus.
