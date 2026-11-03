#!/usr/bin/env bash
# Download Whisper small.en (sherpa-onnx) and Silero VAD into ./models
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models && cd models
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
if [ ! -d sherpa-onnx-whisper-small.en ]; then
  curl -fSL -o whisper.tar.bz2 "$BASE/sherpa-onnx-whisper-small.en.tar.bz2"
  tar -xjf whisper.tar.bz2 && rm whisper.tar.bz2
fi
[ -f silero_vad.onnx ] || curl -fSL -o silero_vad.onnx "$BASE/silero_vad.onnx"
echo "models ready in $(pwd)"
