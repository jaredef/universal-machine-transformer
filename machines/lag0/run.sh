#!/usr/bin/env sh
# lag0 verification ladder.
#  L1  agreement: for every .tm0 fixture (tm0's and lag0's own), the lagging-window decoder
#      reaches the same verdict, the same number of Turing steps, and the same tape as tm0.
#  L2  configurations: the transcript contains the Turing machine's configurations, one per pass.
#  L3  the removal test: the same rule table under a recency window never computes.
set -eu
cd "$(dirname "$0")"

fail() { printf 'FAIL %s\n' "$1"; exit 1; }

for f in ../tm0/fixtures/*.tm0 fixtures/*.tm0; do
  set +e
  want="$(python3 ../tm0/tm0.py "$f" | cut -d' ' -f1-3)"
  got="$(python3 derived/lag0.py "$f" | cut -d' ' -f1-3)"
  set -e
  test "$want" = "$got" || fail "L1 $f: tm0 [$want] lag0 [$got]"
  printf 'L1 agree   %-34s %s\n' "$(basename "$f")" "$got"
done

passes="$(python3 derived/lag0.py ../tm0/fixtures/hex-nibble-a.tm0 --passes)"
for cfg in "pass=0   # q0:A" "pass=1   # 1 q1:_" "pass=2   # 1 0 q2:_" "pass=3   # 1 0 1 q3:_" "pass=4   # 1 0 1 qa:0"; do
  printf '%s\n' "$passes" | grep -qF "$cfg" || fail "L2 missing configuration: $cfg"
done
printf 'L2 configurations of A -> 1010 found in the transcript, one per pass\n'

for f in ../tm0/fixtures/*.tm0 fixtures/*.tm0; do
  set +e
  got="$(python3 derived/lag0.py "$f" --discipline recency)"
  set -e
  case "$got" in
    CYCLE*|NO-RULE*) printf 'L3 recency %-34s %s\n' "$(basename "$f")" "$(printf '%s' "$got" | cut -d' ' -f1)" ;;
    *) fail "L3 $f computed under a recency window: $got" ;;
  esac
done
