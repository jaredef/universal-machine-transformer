#!/usr/bin/env sh
# Run every witness in order: tm0, lag0, lag1, lag2, lag3, lag4, lag5. Pure Python, standard library only; about eight
# minutes (lag3 and lag4 recompute the trained network in pure Python). Stops at the first failure.
set -eu
cd "$(dirname "$0")"
for m in tm0 lag0 lag1 lag2 lag3 lag4 lag5; do
  printf '\n== %s ==\n' "$m"
  sh "machines/$m/run.sh"
done
printf '\nall witnesses green\n'
