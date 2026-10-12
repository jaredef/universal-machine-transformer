#!/usr/bin/env sh
# lag5: lag3's trained network runs the busy beaver champions, BB(2) and BB(4), configuration by configuration.
# Torch-free; reuses lag3's committed weights. About half a minute in pure Python. Fails closed.
set -eu
cd "$(dirname "$0")"
${PYTHON:-python3} derived/bb.py > /dev/null
${PYTHON:-python3} derived/lag5.py
printf 'lag5: all rungs green\n'
