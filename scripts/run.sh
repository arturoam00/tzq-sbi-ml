#!/bin/bash

micromamba activate ml

CONFIG="$1"

python workers/worker.py "$CONFIG"