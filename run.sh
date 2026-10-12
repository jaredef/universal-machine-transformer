#!/usr/bin/env sh
# Run every witness in order: tm0, lag0, lag1, lag2, lag3. Pure Python, standard library only; about four minutes
# (lag3 recomputes its trained network in pure Python). Stops at the first failure.
set -eu
cd "$(dirname "$0")"
for m in tm0 lag0 lag1 lag2 lag3; do
  printf '\n== %s ==\n' "$m"
  sh "machines/$m/run.sh"
done
printf '\nall witnesses green\n'
