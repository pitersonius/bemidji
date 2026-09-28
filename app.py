# this is the backend for the LLMs that are used to analyze the json to later produce the Key moments. DO NOT TOUCH ANYTHING


from __future__ import annotations 

import json
import os
import re
import subprocess
import urllib.error
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

from models import LANGUAGE_NAMES, LLMManager, Settings, WhisperManager, _http_json, find_tool, lms_path

BASE_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = BASE_DIR / "recordings"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
STATIC_DIR = BASE_DIR / "static"

MODELS_DIR = BASE_DIR / "models"

LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))
# transcripts longer than this are summarized in chunks, then merged. scaled up with the
# LM studio context length. this is the size used for other servers, whose context is unknown.
CHUNK_CHARS = int(os.environ.get("LLM_CHUNK_CHARS", "6000"))

# Settings chosen in the Models panel persist in settings.json; env vars only set first-run defaults.
settings = Settings(BASE_DIR / "settings.json", {
    "whisper_model": Path(os.environ.get("WHISPER_MODEL", "ggml-small.bin")).name,
    "whisper_language": os.environ.get("WHISPER_LANGUAGE", "auto"),
    "whisper_on": False,
    # First run: LM Studio if it's installed, otherwise an OpenAI-compatible server (Ollama's default URL).
    "llm_provider": "custom" if os.environ.get("LLM_BASE_URL") or not lms_path() else "lmstudio",
    "llm_model": os.environ.get("LLM_MODEL", ""),  # empty = first LLM downloaded in LM Studio
    "llm_context_length": 8192,
    "custom_base_url": os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
    "custom_model": os.environ.get("LLM_MODEL", ""),
    "summary_language": "same",  # "same" = the recording's language, "en" = always English
})
whisper = WhisperManager(MODELS_DIR, settings, BASE_DIR / "whisper-server.log")
llm = LLMManager(settings)

ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")

for d in (RECORDINGS_DIR, TRANSCRIPTS_DIR):
    d.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# --------------------------------------------------------------------------- helpers


def _check_id(rec_id: str) -> str:
    if not ID_RE.match(rec_id):
        abort(400, "bad recording id")
    return rec_id


def _paths(rec_id: str) -> dict:
    return {
        "wav": RECORDINGS_DIR / f"{rec_id}.wav",
        "whisper_base": TRANSCRIPTS_DIR / f"{rec_id}.whisper",
        "whisper_json": TRANSCRIPTS_DIR / f"{rec_id}.whisper.json",
        "transcript": TRANSCRIPTS_DIR / f"{rec_id}.txt",
        "summary": TRANSCRIPTS_DIR / f"{rec_id}.summary.txt",
        "meta": TRANSCRIPTS_DIR / f"{rec_id}.json",
    }


def _fmt_ts(ms: int) -> str:
    s = ms // 1000
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _load_meta(rec_id: str) -> dict:
    p = _paths(rec_id)["meta"]
    return json.loads(p.read_text()) if p.exists() else {}


def _save_meta(rec_id: str, meta: dict) -> None:
    _paths(rec_id)["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _to_wav(src: Path, dst: Path) -> float:
    """Convert any audio to 16 kHz mono PCM WAV (what whisper.cpp wants). Returns duration in s."""
    ffmpeg = find_tool("ffmpeg", "FFMPEG_BIN")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed. Run ./run.sh to set it up.")
    subprocess.run(
        [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(src),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True, text=True,
    )
    return dst.stat().st_size / (16000 * 2)  # 16-bit mono samples, header is negligible


# --------------------------------------------------------------------------- LLM


def _chat(endpoint: tuple[str, str], system: str, user: str) -> str:
    base_url, model = endpoint
    resp = _http_json(
        f"{base_url}/chat/completions",
        {
            "model": model,
            "temperature": 0.2,
            "max_tokens": 700,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
        timeout=LLM_TIMEOUT,
    )
    return resp["choices"][0]["message"]["content"].strip()


SYSTEM_PROMPT = (
    "You summarize voice-recording transcripts. Reply ONLY with a bullet list: "
    "3 to 7 lines, each starting with '- '. Each bullet is one short sentence describing an "
    "important moment (a decision, fact, idea, question, task or turning point); skip greetings "
    "and small talk. "
    "When the transcript has [mm:ss] timestamps, begin each bullet with the timestamp of the "
    "moment in square brackets, e.g. '- [01:23] ...'. "
    "{language_rule} No intro, no conclusion."
)


def _system_prompt(language: str) -> str:
    """`language` is the recording's detected language code."""
    name = LANGUAGE_NAMES.get(language)
    if settings.get("summary_language") == "en":
        rule = "Write the bullets in English, even if the transcript is in another language."
    elif name:
        # Naming the language works much better with small models than "the transcript's language".
        rule = (f"The transcript is in {name}. Write the bullets only in {name}: no English, "
                "no translations in brackets. Summarize in your own words instead of copying lines.")
    else:
        rule = "Write the bullets in the same language as the transcript."
    return SYSTEM_PROMPT.format(language_rule=rule)


def _chunks(lines: list[str], limit: int) -> list[str]:
    out, cur, size = [], [], 0
    for line in lines:
        if cur and size + len(line) > limit:
            out.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def _parse_bullets(text: str) -> list[str]:
    bullets = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", line)
        if m and m.group(1).strip():
            bullets.append(m.group(1).strip())
    if not bullets and text.strip():  # model ignored the format; keep its non-empty lines
        bullets = [l.strip() for l in text.splitlines() if l.strip()]
    return [_normalize_ts(b) for b in bullets[:10]]


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) > 1 and text[0] in "\"“'«" and text[-1] in "\"”'»":
        return text[1:-1].strip()
    return text


def _normalize_ts(bullet: str) -> str:
    """Rewrite a leading timestamp like '[0:4]' or '00:07 -' as '[00:04]'."""
    m = re.match(r"^\[?(?:(\d{1,2}):)?(\d{1,2}):(\d{1,2})\]?\s*[-–—:]?\s*(.*)$", bullet)
    if not m:
        return _unquote(bullet)
    h, mnt, sec, rest = int(m.group(1) or 0), int(m.group(2)), int(m.group(3)), m.group(4)
    return f"[{_fmt_ts((h * 3600 + mnt * 60 + sec) * 1000)}] {_unquote(rest)}"


def _summarize(segments: list[dict], endpoint: tuple[str, str], language: str) -> list[str]:
    lines = [f"[{_fmt_ts(s['start'])}] {s['text']}" for s in segments]
    chunk_chars = CHUNK_CHARS
    if settings.get("llm_provider") == "lmstudio":
        # ~4 chars per token; leave room for the prompt and the answer.
        chunk_chars = max(CHUNK_CHARS, int(settings.get("llm_context_length") * 2.2))
    parts = _chunks(lines, chunk_chars)
    if len(parts) == 1:
        return _parse_bullets(_chat(endpoint, _system_prompt(language), f"Transcript:\n{parts[0]}"))
    # Map-reduce for long recordings: bullets per chunk, then merge into the final list.
    partial = []
    for i, part in enumerate(parts, 1):
        partial += _parse_bullets(
            _chat(endpoint, _system_prompt(language), f"Transcript (part {i} of {len(parts)}):\n{part}")
        )
    merged = "\n".join(f"- {b}" for b in partial)
    return _parse_bullets(_chat(
        endpoint, _system_prompt(language),
        "These are key moments extracted from consecutive parts of one recording. "
        "Merge them into the single most important 3-7 moments, keeping timestamps:\n" + merged,
    ))


def _run_summary(rec_id: str) -> dict:
    p = _paths(rec_id)
    meta = _load_meta(rec_id)
    segments = meta.get("segments", [])
    endpoint = llm.endpoint() if segments else None
    if segments and not endpoint:
        # The LLM is switched off: leave the recording un-summarized and tell the UI why.
        return {**meta, "llm_off": True}
    if not segments:
        meta.update(bullets=[], summary_error=None)
    else:
        try:
            meta["bullets"] = _summarize(segments, endpoint, meta.get("language", ""))
            meta["summary_error"] = None
            meta["summary_model"] = endpoint[1]
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            meta["bullets"] = []
            meta["summary_error"] = (
                f"Could not reach the local LLM at {endpoint[0]} ({getattr(e, 'reason', e)}). "
                "Check it under Models and click Retry."
            )
        except Exception as e:  # noqa: BLE001 - surface any LLM failure to the UI
            meta["bullets"] = []
            meta["summary_error"] = f"Summary failed: {e}"
    if meta["bullets"]:
        p["summary"].write_text(
            f"Key moments — {meta.get('title', rec_id)}\n\n"
            + "\n".join(f"• {b}" for b in meta["bullets"]) + "\n"
        )
    _save_meta(rec_id, meta)
    return meta


# --------------------------------------------------------------------------- routes


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/models")
def models_status():
    return jsonify(whisper=whisper.status(), llm=llm.status(), ffmpeg=find_tool("ffmpeg", "FFMPEG_BIN") is not None)


def _model_action(fn):
    try:
        fn()
    except (ValueError, RuntimeError) as e:
        return jsonify(error=str(e)), 400
    return models_status()


@app.post("/api/models/whisper")
def whisper_update():
    body = request.get_json(force=True)

    def apply():
        if "language" in body:
            whisper.set_language(str(body["language"]) or "auto")
        if "model" in body:
            whisper.set_model(body["model"])
        if "on" in body:
            whisper.set_on(bool(body["on"]))
    return _model_action(apply)


@app.post("/api/models/whisper/<name>/download")
def whisper_download(name):
    return _model_action(lambda: whisper.download(name))


@app.delete("/api/models/whisper/<name>/download")
def whisper_cancel_download(name):
    return _model_action(lambda: whisper.cancel_download(name))


@app.delete("/api/models/whisper/<name>")
def whisper_delete(name):
    return _model_action(lambda: whisper.delete(name))


@app.post("/api/models/llm")
def llm_update():
    body = request.get_json(force=True)

    def apply():
        if "provider" in body:
            if body["provider"] not in ("lmstudio", "custom"):
                raise ValueError("unknown provider")
            settings.update(llm_provider=body["provider"])
        if "base_url" in body:
            url = str(body["base_url"]).strip().rstrip("/")
            if not url.startswith(("http://", "https://")):
                raise ValueError("The server URL must start with http:// or https://")
            settings.update(custom_base_url=url)
        if "summary_language" in body:
            if body["summary_language"] not in ("same", "en"):
                raise ValueError("summary_language must be 'same' or 'en'")
            settings.update(summary_language=body["summary_language"])
        if "context_length" in body:
            llm.set_context_length(int(body["context_length"]))
        if "model" in body:
            llm.set_model(str(body["model"]))
        if "on" in body:
            llm.set_on(bool(body["on"]))
    return _model_action(apply)


@app.post("/api/transcribe")
def transcribe():
    f = request.files.get("audio")
    if not f:
        abort(400, "no audio uploaded")

    now = datetime.now()
    rec_id = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    p = _paths(rec_id)
    source = "mic" if request.form.get("source") == "mic" else "file"
    ext = Path(f.filename or "").suffix.lower() or ".webm"
    raw = RECORDINGS_DIR / f"{rec_id}.src{ext}"  # distinct from the .wav ffmpeg writes
    f.save(raw)

    try:
        duration = _to_wav(raw, p["wav"])
        segments, language, mode = whisper.transcribe(p["wav"], p["whisper_base"])
    except subprocess.CalledProcessError as e:
        return jsonify(error=f"ffmpeg could not read the audio: {e.stderr.strip()[-500:]}"), 500
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 500
    finally:
        raw.unlink(missing_ok=True)  # keep only the normalized WAV
        p["whisper_json"].unlink(missing_ok=True)

    if request.form.get("title"):
        title = request.form["title"]
    elif source == "file" and f.filename:
        title = Path(f.filename).stem
    else:
        title = f"Recording {now:%Y-%m-%d %H:%M}"
    plain = " ".join(s["text"] for s in segments)
    p["transcript"].write_text(
        f"{title}\nRecorded: {now:%Y-%m-%d %H:%M:%S}   Duration: {_fmt_ts(int(duration * 1000))}"
        f"   Language: {language}\n\n"
        + "\n".join(f"[{_fmt_ts(s['start'])}] {s['text']}" for s in segments)
        + "\n\n--- Plain text ---\n" + plain + "\n"
    )
    meta = {
        "id": rec_id, "title": title, "created": now.isoformat(timespec="seconds"),
        "duration": round(duration, 1), "language": language,
        "source": source, "original_name": f.filename if source == "file" else None,
        "whisper_model": settings.get("whisper_model"), "whisper_mode": mode,
        "segments": segments, "bullets": None, "summary_error": None,
    }
    _save_meta(rec_id, meta)
    return jsonify(meta)


@app.post("/api/recordings/<rec_id>/summarize")
def summarize(rec_id):
    _check_id(rec_id)
    if not _paths(rec_id)["meta"].exists():
        abort(404)
    return jsonify(_run_summary(rec_id))


@app.get("/api/recordings")
def list_recordings():
    items = []
    for meta_path in sorted(TRANSCRIPTS_DIR.glob("*.json"), reverse=True):
        if meta_path.name.endswith(".whisper.json"):
            continue
        m = json.loads(meta_path.read_text())
        items.append({k: m.get(k) for k in ("id", "title", "created", "duration", "language", "source")})
    return jsonify(items)


@app.get("/api/recordings/<rec_id>")
def get_recording(rec_id):
    _check_id(rec_id)
    meta = _load_meta(rec_id)
    if not meta:
        abort(404)
    return jsonify(meta)


@app.delete("/api/recordings/<rec_id>")
def delete_recording(rec_id):
    _check_id(rec_id)
    p = _paths(rec_id)
    for key in ("wav", "transcript", "summary", "meta", "whisper_json"):
        p[key].unlink(missing_ok=True)
    return jsonify(ok=True)


@app.get("/files/<rec_id>/<kind>")
def download(rec_id, kind):
    _check_id(rec_id)
    p = _paths(rec_id)
    targets = {"transcript": p["transcript"], "summary": p["summary"], "audio": p["wav"]}
    if kind not in targets or not targets[kind].exists():
        abort(404)
    path = targets[kind]
    return send_from_directory(path.parent, path.name, as_attachment=request.args.get("dl") == "1")


if __name__ == "__main__":
    whisper.restore()
    port = int(os.environ.get("PORT", "5050"))
    print(f"Voice notes running at http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
