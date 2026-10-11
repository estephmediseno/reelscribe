reelscribe
==========
reelscribe is a command-line tool that downloads batches of TikTok and Instagram
Reels videos, transcribes speech locally with `faster-whisper`, writes separate
output files per video (`.md`, `.srt`, and `.json`), and generates a consolidated
analytical summary using a local LLM via Ollama.

How it works
------------
1. **Short-link resolution and deduplication**: Resolves mobile short links (`vm.tiktok.com`, `vt.tiktok.com`, `instagr.am`), strips tracking query parameters (`?_r=1&_t=...`), and skips duplicate canonical URLs in the same batch.
2. **Audio and metadata extraction**: `yt-dlp` fetches video metadata (author, platform, ID, title, duration) and extracts the audio stream.
3. **GPU-accelerated VAD transcription**: `faster-whisper` runs locally with automatic NVIDIA CUDA detection (`float16`, with automatic CPU `int8` fallback), batch model caching, and Voice Activity Detection (`vad_filter=True`) to ignore background music and silence.
4. **Per-video outputs**: Writes three separate files per video in `transcriptions/`:
   * `NN_<platform>_<author>_<id>.md`: Metadata header, clean full-text transcript, and timestamped breakdown.
   * `NN_<platform>_<author>_<id>.srt`: Standard SubRip subtitle file.
   * `NN_<platform>_<author>_<id>.json`: Structured `yt-dlp` metadata, full text, segments, and word-level timestamps.
5. **Local Ollama summarization**: Checks if local Ollama is reachable, bundles all transcripts in the batch, and queries a local model (`Qwen2.5-14B-Instruct` by default) to write `transcriptions/resumen.md`.

Requirements
------------
* Python 3.10+
* `ffmpeg`
* `yt-dlp`
* [Ollama](https://ollama.com/) running locally (optional; automatically skipped if offline or `--no-summary` is passed)

Installation
------------
```sh
git clone https://github.com/riccivr/reelscribe.git
cd reelscribe
python3 -m venv .venv
.venv/bin/pip install -e .
```

Usage
-----
```sh
reelscribe [-f file] [-o dir] [-m model] [-l lang] [-d auto|cuda|cpu]
           [--compute-type type] [--ollama-model model] [--ollama-url url]
           [--no-summary] [-c cookies.txt] [--cookies-from-browser browser]
           [--keep-audio] [--open-dir] <url...>
```

### Options
* `-f, --file`: Text file containing one URL per line (also reads URLs from `stdin` when piped).
* `-o, --output-dir`: Output directory for generated files (default: `transcriptions/`).
* `-m, --model`: `faster-whisper` model size: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` (default: `small`).
* `-l, --language`: Force source language code (e.g. `es`, `en`). Auto-detected by default.
* `-d, --device`: Device for `faster-whisper`: `auto`, `cuda`, `cpu` (default: `auto`, uses NVIDIA GPU if available).
* `--compute-type`: Compute precision for `faster-whisper` (e.g. `float16`, `int8_float16`, `int8`). Auto-selected by device if omitted.
* `--ollama-model`: Local Ollama model name for batch summarization.
* `--ollama-url`: Ollama chat API URL (default: `http://localhost:11434/api/chat`).
* `--no-summary`: Skip the final Ollama summary step.
* `-c, --cookies`: Path to a `cookies.txt` file for authenticated video downloads.
* `--cookies-from-browser`: Browser name to extract cookies from when required by Instagram or TikTok.
* `--keep-audio`: Keep the extracted audio file alongside the transcripts.
* `--open-dir`: Open the output directory in File Explorer upon completion.

Examples
--------
Transcribe multiple URLs (including TikTok short links) and generate a batch summary:
```sh
reelscribe "https://vm.tiktok.com/ZN8BDuYDp/" "https://www.instagram.com/reel/EXAMPLE/"
```

Process a batch of URLs from a text file into a custom folder:
```sh
reelscribe -f urls.txt -o transcriptions/client_october
```
