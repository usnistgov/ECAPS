#!/bin/bash

set -e

rm -f 5b.out

nohup python 5b-ECAPS_classification-AlgorithmSelection.py >> 5b.out 2>&1 & echo $! > run_5b.pid

#nohup python script_example.py >>py.out 2>&1 &

# `jobs` shows running subprocesses (in that shell)
# `fg` brings the last running subprocess into the foreground (this shell)

