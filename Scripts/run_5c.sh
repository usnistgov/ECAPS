#!/bin/bash

set -e

rm -f 5c.out

nohup python 5c-ECAPS_classification-LSVC_LOGO.py >> 5c.out 2>&1 & echo $! > run_5c.pid

#nohup python script_example.py >>py.out 2>&1 &

# `jobs` shows running subprocesses (in that shell)
# `fg` brings the last running subprocess into the foreground (this shell)

