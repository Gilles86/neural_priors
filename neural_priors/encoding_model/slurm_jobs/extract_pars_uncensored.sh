#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for MODEL in 0 1 2 3 4 5 14 15 18 31 32 33 34 35; do
    echo "=== Model ${MODEL} (groundtruth) ==="
    python "${SCRIPT_DIR}/extract_pars.py" ${MODEL} --smoothed

    echo "=== Model ${MODEL} (fit_responses) ==="
    python "${SCRIPT_DIR}/extract_pars.py" ${MODEL} --smoothed --fit_responses
done
