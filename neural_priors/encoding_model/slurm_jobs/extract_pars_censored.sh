#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for MODEL in 3 5; do
    echo "=== Model ${MODEL} (groundtruth, censored) ==="
    python "${SCRIPT_DIR}/extract_pars.py" ${MODEL} --smoothed --censored

    echo "=== Model ${MODEL} (fit_responses, censored) ==="
    python "${SCRIPT_DIR}/extract_pars.py" ${MODEL} --smoothed --censored --fit_responses
done
