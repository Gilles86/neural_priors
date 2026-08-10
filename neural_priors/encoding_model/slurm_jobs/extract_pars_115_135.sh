#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for MODEL in {115..135}; do
    echo "=== Model ${MODEL} (groundtruth) ==="
    python "${SCRIPT_DIR}/extract_pars.py" ${MODEL} --smoothed

    echo "=== Model ${MODEL} (fit_responses) ==="
    python "${SCRIPT_DIR}/extract_pars.py" ${MODEL} --smoothed --fit_responses
done
