#!/bin/bash

set -e

rm 5a.out

nohup python Scripts/5a-ECAPS_classification-PreProcessingSelection.py >> 5a.out 2>&1 & echo $! > run_5a.pid

#nohup python script_example.py >>py.out 2>&1 &

# `jobs` shows running subprocesses (in that shell)
# `fg` brings the last running subprocess into the foreground (this shell)

