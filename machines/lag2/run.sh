#!/usr/bin/env sh
# lag2: a TRAINED transformer as the step of the lagging-window decoder. Torch-free witness over the committed
# weights (fixtures/lag2_training.json, produced by fixtures/train_lag2.ref.py with torch). Fails closed.
set -eu
cd "$(dirname "$0")"
${PYTHON:-python3} derived/lag2.py all
printf 'lag2: all rungs green\n'
