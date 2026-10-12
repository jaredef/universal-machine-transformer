#!/usr/bin/env sh
# lag4: lag3's trained network draws cellular automata (Rules 90, 30, 110), running a program it never saw.
# Torch-free; reuses lag3's committed weights. About three and a half minutes in pure Python. Fails closed.
set -eu
cd "$(dirname "$0")"
${PYTHON:-python3} derived/lag4.py
printf 'lag4: all rungs green\n'
