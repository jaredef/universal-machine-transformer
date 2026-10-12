#!/usr/bin/env sh
# lag1 verification ladder: lag0's rule table realized as transformer weights.
#  LT-1  the forward pass equals lag0's rule table on every window the decoder can meet (exhaustive)
#  LT-2  decoding with the forward pass agrees with tm0 on every fixture: verdict, steps, and tape
#  LT-3  the same weights under a recency window never compute
#  LT-4  MLP 1 has exactly one neuron per row of the Turing rule table, per slot
# Deterministic, standard library only, exact integer and Fraction arithmetic. Fails closed.
set -eu
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
fail() { printf 'FAIL %s\n' "$1" >&2; exit 1; }

for f in ../tm0/fixtures/*.tm0 ../lag0/fixtures/*.tm0; do
  name=$(basename "$f")
  out=$($PY derived/lag1.py exhaustive "$f") || fail "LT-1 $name: $out"
  case "$out" in FORWARD-EQUALS-TABLE*mismatches=0) ;; *) fail "LT-1 $name: $out" ;; esac
  set +e; want=$($PY ../tm0/tm0.py "$f" | cut -d' ' -f1-3); set -e
  got=$($PY derived/lag1.py decode "$f")
  test "$want" = "$got" || fail "LT-2 $name: tm0 [$want] lag1 [$got]"
  rec=$($PY derived/lag1.py recency "$f")
  case "$rec" in CYCLE*|NO-RULE*) ;; *) fail "LT-3 $name computed under a recency window: $rec" ;; esac
  neu=$($PY derived/lag1.py neurons "$f")
  case "$neu" in RULE-TABLE-IS-NEURONS*) ;; *) fail "LT-4 $name: $neu" ;; esac
  printf '%-30s LT-1 %-42s LT-2 %-26s LT-3 %-8s LT-4 %s\n' "$name" \
    "$(printf '%s' "$out" | cut -d' ' -f2,3)" "$got" "$(printf '%s' "$rec" | cut -d' ' -f1)" \
    "$(printf '%s' "$neu" | cut -d' ' -f2,3)"
done
printf '\nweights for binary increment: %s\n' "$($PY derived/lag1.py params ../lag0/fixtures/binary-increment.tm0)"
printf 'lag1: all rungs green\n'
