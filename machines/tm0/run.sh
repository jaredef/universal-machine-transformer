#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

check_accept() {
  name="$1"
  expected="$2"
  output="$(python3 tm0.py "fixtures/$name.tm0")"
  test "$output" = "$expected"
  printf '%s\n' "$output"
}

check_reject() {
  name="$1"
  expected="$2"
  set +e
  output="$(python3 tm0.py "fixtures/$name.tm0")"
  status="$?"
  set -e
  test "$status" = 1
  test "$output" = "$expected"
  printf '%s\n' "$output"
}

check_accept write-one "ACCEPT steps=1 tape=1"
check_accept scan-right "ACCEPT steps=4 tape=111"
check_reject missing-transition "REJECT steps=0 tape=1"
check_accept blank-default "ACCEPT steps=2 tape=11"
check_accept hex-nibble-a "ACCEPT steps=4 tape=1010"
