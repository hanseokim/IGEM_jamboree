# IGEM_jamboree

Pipeline that extracts English speech (transcripts) from the judging-session recordings in `Judgeinput/`,
using [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with Silero VAD and Whisper `small.en`.

```
audio --ffmpeg--> 16 kHz mono wav --Silero VAD--> speech segments --Whisper small.en--> .txt / .srt / .json
```

## Usage
```bash
pip install -r requirements.txt   # needs ffmpeg on PATH
scripts/run_all.sh                # downloads models (~1 GB, git-ignored) and processes all of Judgeinput/
```
Results are written to `Judgeoutput/<name>.{txt,srt,json}`. Files already processed are skipped.
Single file / other folder: `PYTHONPATH=. python3 -m src.jamboree_qa.pipeline <path> -o <outdir>`
