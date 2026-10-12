#!/usr/bin/env sh
# lag3: one trained network runs any machine, by learning the steps of a universal Turing machine.
# Torch-free witness over the committed weights (fixtures/lag3_training.json, from fixtures/train_lag3.ref.py).
# About two minutes in pure Python. Fails closed.
set -eu
cd "$(dirname "$0")"
${PYTHON:-python3} derived/utm.py
${PYTHON:-python3} derived/lag3w.py
printf 'lag3: all rungs green\n'
