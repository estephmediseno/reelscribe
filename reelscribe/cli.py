#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

SHORT_LINK_HOSTS = {"vm.tiktok.com", "vt.tiktok.com", "instagr.am"}
DEFAULT_OLLAMA_MODEL = "hf.co/mradermacher/Qwen2.5-14B-Instruct-abliterated-v2-GGUF:Q4_K_M"


def slugify(text: str, max_len: int = 45) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text[:max_len] or "video"


def format_srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1_000
    ms %= 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_mmss(seconds: float) -> str:
    total_secs = int(round(seconds))
    mins, secs = divmod(total_secs, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def resolve_url(url: str) -> str:
    """Resolve mobile short links (e.g. vm.tiktok.com, vt.tiktok.com) and strip tracking parameters."""
    url = url.strip()
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if host in SHORT_LINK_HOSTS or "/t/" in parsed.path:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                url = resp.geturl()
                parsed = urlparse(url)
                host = parsed.netloc.lower()
        except Exception:
            pass

    if "tiktok.com" in host or "instagram.com" in host:
        parsed = parsed._replace(query="", fragment="")
        url = urlunparse(parsed)

    return url


def download_audio(url: str, dest_dir: Path, cookies_from_browser: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Download audio stream and metadata from a TikTok or Instagram URL using yt-dlp."""
    resolved_url = resolve_url(url)
    out_tmpl = str(dest_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--write-info-json",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "-o",
        out_tmpl,
    ]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    cmd.append(resolved_url)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed for {url}:\n{proc.stderr.strip()}")

    info_files = sorted(dest_dir.glob("*.info.json"), key=os.path.getmtime, reverse=True)
    if not info_files:
        raise FileNotFoundError(f"No .info.json file found after downloading {url}")

    info_path = info_files[0]
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    audio_files = [
        p for p in dest_dir.iterdir()
        if p.is_file() and not p.name.endswith(".json") and p.stem == info_path.name.removesuffix(".info.json")
    ]
    if not audio_files:
        audio_files = [p for p in dest_dir.glob("*.mp3")]
    if not audio_files:
        raise FileNotFoundError(f"Audio file not found for {url}")

    return audio_files[0], info


def transcribe_audio(
    audio_path: Path,
    model_size: str = "small",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
) -> dict[str, Any]:
    """Transcribe an audio file with faster-whisper, returning segments and word-level timestamps."""
    from faster_whisper import WhisperModel

    def _run(dev: str, ctype: str):
        model = WhisperModel(model_size, device=dev, compute_type=ctype)
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )
        return list(segments_gen), info

    try:
        segments_list, info = _run(device, compute_type)
    except Exception:
        segments_list, info = _run("cpu", "int8")

    segments = []
    full_text_parts = []
    for idx, seg in enumerate(segments_list, start=1):
        text = seg.text.strip()
        if not text:
            continue
        words = []
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3),
                    }
                )
        segments.append(
            {
                "id": idx,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": text,
                "words": words,
            }
        )
        full_text_parts.append(text)

    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "full_text": " ".join(full_text_parts),
        "segments": segments,
    }


def write_outputs(
    index: int,
    url: str,
    info: dict[str, Any],
    transcript: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    """Write per-video .md, .srt, and .json output files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    platform = (info.get("extractor_key") or info.get("extractor") or "video").lower()
    vid_id = str(info.get("id") or f"v{index}")
    title = info.get("title") or info.get("description") or vid_id
    title_first_line = title.splitlines()[0].strip() if title else vid_id
    uploader = info.get("uploader") or info.get("channel") or info.get("creator") or "Unknown"
    duration = info.get("duration") or transcript.get("duration") or 0

    slug = slugify(f"{platform}_{uploader}_{vid_id}")
    base_name = f"{index:02d}_{slug}"

    md_path = out_dir / f"{base_name}.md"
    srt_path = out_dir / f"{base_name}.srt"
    json_path = out_dir / f"{base_name}.json"

    # 1. SubRip (.srt) file
    srt_lines = []
    for i, seg in enumerate(transcript["segments"], start=1):
        start_ts = format_srt_timestamp(seg["start"])
        end_ts = format_srt_timestamp(seg["end"])
        srt_lines.append(f"{i}\n{start_ts} --> {end_ts}\n{seg['text']}\n")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    # 2. Structured JSON (.json) file
    payload = {
        "index": index,
        "url": url,
        "webpage_url": info.get("webpage_url") or url,
        "platform": platform,
        "video_id": vid_id,
        "title": title_first_line,
        "description": info.get("description") or "",
        "uploader": uploader,
        "uploader_id": info.get("uploader_id") or "",
        "upload_date": info.get("upload_date") or "",
        "duration": duration,
        "language": transcript["language"],
        "language_probability": transcript["language_probability"],
        "full_text": transcript["full_text"],
        "segments": transcript["segments"],
        "transcribed_at": datetime.now().isoformat(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. Clean Markdown (.md) file
    timestamped_lines = [
        f"- `[{format_mmss(seg['start'])} - {format_mmss(seg['end'])}]` {seg['text']}"
        for seg in transcript["segments"]
    ]
    timestamped_block = "\n".join(timestamped_lines) if timestamped_lines else "_No speech detected_"

    md_content = f"""# {index:02d}. {title_first_line}

- **URL:** {url}
- **Plataforma:** {platform.capitalize()}
- **Autor:** {uploader}
- **ID:** `{vid_id}`
- **Duración:** {format_mmss(float(duration))} ({duration}s)
- **Idioma detectado:** `{transcript['language']}` ({transcript['language_probability']:.0%})

## Transcripción limpia

{transcript['full_text'] or '_Sin voz detectada_'}

## Transcripción con marcas de tiempo

{timestamped_block}
"""
    md_path.write_text(md_content, encoding="utf-8")

    return {"md": md_path, "srt": srt_path, "json": json_path}


def summarize_with_ollama(
    items: list[dict[str, Any]],
    out_dir: Path,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = "http://localhost:11434/api/chat",
) -> Path | None:
    """Generate a batch summary across all transcripts using local Ollama and write resumen.md."""
    if not items:
        return None

    context_blocks = []
    for item in items:
        context_blocks.append(
            f"### Video {item['index']}: {item['title']}\n"
            f"- Autor: {item['uploader']} ({item['platform']})\n"
            f"- URL: {item['url']}\n"
            f"- Transcripción:\n{item['full_text'] or '(Sin voz detectada)'}\n"
        )
    joined_transcripts = "\n---\n".join(context_blocks)

    system_prompt = (
        "Eres un analista directo y claro. Recibirás las transcripciones completas de varios videos "
        "de TikTok o Instagram. Redacta en español:\n"
        "1. Un resumen ejecutivo general que conecte los puntos clave de todos los videos.\n"
        "2. Puntos principales de cada video por separado.\n"
        "3. Conclusiones o ideas accionables.\n"
        "Usa un estilo directo, sin relleno ni frases típicas de IA."
    )

    payload = json.dumps(
        {
            "model": ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": joined_transcripts},
            ],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        ollama_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        summary_text = data.get("message", {}).get("content", "").strip()

    resumen_path = out_dir / "resumen.md"
    header = (
        f"# Resumen general ({len(items)} videos)\n\n"
        f"- **Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"- **Modelo local (Ollama):** `{ollama_model}`\n\n"
    )
    resumen_path.write_text(header + summary_text + "\n", encoding="utf-8")
    return resumen_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and transcribe TikTok/Instagram videos (.md, .srt, .json) and summarize with local Ollama."
    )
    parser.add_argument("urls", nargs="*", help="TikTok, Instagram, or video URLs (including short links).")
    parser.add_argument("-f", "--file", type=Path, help="Text file containing one URL per line.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("transcriptions"),
        help="Output directory (default: transcriptions/).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="small",
        help="faster-whisper model size (tiny, base, small, medium, large-v3, large-v3-turbo). Default: small.",
    )
    parser.add_argument("-l", "--language", default=None, help="Force language code (e.g. es, en). Auto by default.")
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Local Ollama model for batch summary (default: {DEFAULT_OLLAMA_MODEL}).",
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip the final Ollama summary step.")
    parser.add_argument("--cookies-from-browser", default=None, help="Browser name to extract cookies from if needed.")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the extracted .mp3 file in the output directory.")

    args = parser.parse_args()

    raw_urls: list[str] = list(args.urls)
    if args.file and args.file.exists():
        for line in args.file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                raw_urls.append(line)

    if not raw_urls:
        parser.print_help()
        return 1

    urls: list[str] = []
    seen_canonical: set[str] = set()
    for raw_url in raw_urls:
        canonical = resolve_url(raw_url)
        if canonical in seen_canonical:
            print(f"[Skipped duplicate] {raw_url} -> {canonical}", file=sys.stderr)
            continue
        seen_canonical.add(canonical)
        urls.append(canonical)

    results = []
    summary_inputs = []
    errors = []

    for idx, url in enumerate(urls, start=1):
        print(f"[{idx}/{len(urls)}] Processing: {url}", file=sys.stderr)
        try:
            with tempfile.TemporaryDirectory(prefix="reelscribe_") as tmp_str:
                tmp_dir = Path(tmp_str)
                audio_path, info = download_audio(url, tmp_dir, args.cookies_from_browser)
                transcript = transcribe_audio(
                    audio_path,
                    model_size=args.model,
                    language=args.language,
                )
                paths = write_outputs(idx, url, info, transcript, args.output_dir)
                if args.keep_audio:
                    target_audio = args.output_dir / f"{paths['md'].stem}{audio_path.suffix}"
                    target_audio.write_bytes(audio_path.read_bytes())

                platform = (info.get("extractor_key") or info.get("extractor") or "video").lower()
                title = (info.get("title") or info.get("description") or str(info.get("id") or f"v{idx}")).splitlines()[0].strip()
                uploader = info.get("uploader") or info.get("channel") or info.get("creator") or "Unknown"

                summary_inputs.append(
                    {
                        "index": idx,
                        "url": url,
                        "platform": platform,
                        "title": title,
                        "uploader": uploader,
                        "full_text": transcript["full_text"],
                    }
                )
                results.append(
                    {
                        "index": idx,
                        "url": url,
                        "md": str(paths["md"]),
                        "srt": str(paths["srt"]),
                        "json": str(paths["json"]),
                    }
                )
                print(f"  -> Saved: {paths['md'].name} (.md, .srt, .json)", file=sys.stderr)
        except Exception as exc:
            print(f"  [ERROR] {url}: {exc}", file=sys.stderr)
            errors.append({"index": idx, "url": url, "error": str(exc)})

    resumen_file = None
    if summary_inputs and not args.no_summary:
        print(f"Generating batch summary with Ollama ({args.ollama_model})...", file=sys.stderr)
        try:
            resumen_path = summarize_with_ollama(summary_inputs, args.output_dir, ollama_model=args.ollama_model)
            if resumen_path:
                resumen_file = str(resumen_path)
                print(f"  -> Summary saved to: {resumen_path}", file=sys.stderr)
        except Exception as exc:
            print(f"  [ERROR Ollama Summary]: {exc}", file=sys.stderr)
            errors.append({"stage": "ollama_summary", "error": str(exc)})

    summary_manifest = {
        "processed": len(results),
        "failed": len(errors),
        "summary_file": resumen_file,
        "results": results,
        "errors": errors,
    }
    print(json.dumps(summary_manifest, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
