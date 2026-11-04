"""English speech extraction: media -> 16 kHz mono wav -> Silero VAD -> Whisper small.en -> txt/srt/json."""
import argparse, json, subprocess, sys, tempfile, wave
from pathlib import Path

import numpy as np
import sherpa_onnx

SR = 16000
ROOT = Path(__file__).resolve().parents[2]


def extract_audio(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)], check=True)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def make_recognizer(model_dir: Path, threads: int):
    p = model_dir / "sherpa-onnx-whisper-small.en"
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(p / "small.en-encoder.int8.onnx"),
        decoder=str(p / "small.en-decoder.int8.onnx"),
        tokens=str(p / "small.en-tokens.txt"),
        language="en", task="transcribe", num_threads=threads)


def make_vad(model_dir: Path):
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(model_dir / "silero_vad.onnx")
    cfg.silero_vad.min_silence_duration = 0.4
    cfg.silero_vad.max_speech_duration = 25  # Whisper window is 30 s
    cfg.sample_rate = SR
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)


def segments(samples, vad):
    win = 512
    out = []

    def drain():
        while not vad.empty():
            s = vad.front
            out.append((s.start / SR, np.array(s.samples, dtype=np.float32)))
            vad.pop()

    for i in range(0, len(samples) - win + 1, win):
        vad.accept_waveform(samples[i:i + win])
        drain()
    vad.flush()
    drain()
    return out


def ts(t, sep=","):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}{sep}{ms % 1000:03}"


def transcribe(src: Path, model_dir: Path, threads: int = 4):
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        extract_audio(src, wav)
        samples = read_wav(wav)
    rec = make_recognizer(model_dir, threads)
    res = []
    chunk = 28 * SR  # Whisper handles <30 s per decode; split longer VAD segments
    for seg_start, full in segments(samples, make_vad(model_dir)):
        for off in range(0, len(full), chunk):
            seg = full[off:off + chunk]
            start = seg_start + off / SR
            st = rec.create_stream()
            st.accept_waveform(SR, seg)
            rec.decode_stream(st)
            text = st.result.text.strip()
            if text:
                res.append({"start": round(start, 2), "end": round(start + len(seg) / SR, 2), "text": text})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, nargs="?", default=ROOT / "Judgeinput", help="audio/video file or folder (default: Judgeinput)")
    ap.add_argument("-o", "--out-dir", type=Path, default=ROOT / "Judgeoutput")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    exts = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".webm"}
    files = sorted(f for f in a.input.iterdir() if f.suffix.lower() in exts) if a.input.is_dir() else [a.input]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        stem = a.out_dir / f.stem
        if stem.with_suffix(".json").exists():
            print(f"skip (done): {f.name}")
            continue
        print(f"transcribing: {f.name}", flush=True)
        res = transcribe(f, a.models, a.threads)
        stem.with_suffix(".txt").write_text("\n".join(r["text"] for r in res) + "\n")
        stem.with_suffix(".srt").write_text("".join(
            f"{i}\n{ts(r['start'])} --> {ts(r['end'])}\n{r['text']}\n\n" for i, r in enumerate(res, 1)))
        stem.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"  {len(res)} segments -> {stem}.{{txt,srt,json}}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
