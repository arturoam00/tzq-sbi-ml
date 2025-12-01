#!/bin/bash

set -euo pipefail

setup_env() {
    PATH="$ENV:$PATH"
}

setup_env


TARBALL="$1"
tar -xzf "$TARBALL"

python -m workers.worker *.yaml

