#!/usr/bin/env bash
# Download models (if needed) and transcribe every audio file in Judgeinput/ -> Judgeoutput/
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/download_models.sh
PYTHONPATH=. python3 -m src.jamboree_qa.pipeline "${1:-Judgeinput}" -o Judgeoutput
