#!/bin/bash

set -e

rm -f 3b.out

nohup python Scripts/3b-ECAPS_classification-NoTuningSCRIPT.py >> 3b.out 2>&1 & echo $! > run_3b.pid

#nohup python script_example.py >>py.out 2>&1 &

# `jobs` shows running subprocesses (in that shell)
# `fg` brings the last running subprocess into the foreground (this shell)

