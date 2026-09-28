"""Local model management.

* Whisper (whisper.cpp): pick / download / delete ggml models; "on" keeps the model loaded in a
  `whisper-server` process, "off" frees it and falls back to a one-shot `whisper-cli` per file.
* LLM: LM Studio (fully managed through its `lms` CLI: server start/stop, model load/unload)
  or any other OpenAI-compatible server (Ollama, llama-server, …) that you run yourself.
"""

from __future__ import annotations  # allows "str | None" annotations on Python 3.9

import atexit
import json
import os
import shutil
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# run.sh builds whisper.cpp here when it isn't installed system-wide (e.g. on Linux).
VENDOR_BIN = BASE_DIR / "vendor" / "whisper.cpp" / "build" / "bin"


def find_tool(name: str, env_var: str | None = None) -> str | None:
    """Locate an external program: $ENV override, then PATH, then the project's own copies."""
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    found = shutil.which(name)
    if found:
        return found
    local = VENDOR_BIN / name
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    if name == "ffmpeg":
        try:  # pip-installed static ffmpeg, used when no system ffmpeg exists (see run.sh)
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            return None
    return None


def lms_path() -> str | None:
    """LM Studio's CLI; it lives in ~/.lmstudio/bin on both macOS and Linux."""
    found = shutil.which("lms") or str(Path.home() / ".lmstudio/bin/lms")
    return found if Path(found).exists() else None


# --------------------------------------------------------------------------- settings


class Settings:
    """Small JSON-backed settings store."""

    def __init__(self, path: Path, defaults: dict):
        self.path = path
        self._lock = threading.Lock()
        self._data = dict(defaults)
        if path.exists():
            try:
                self._data.update(json.loads(path.read_text()))
            except json.JSONDecodeError:
                pass

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def update(self, **values):
        with self._lock:
            self._data.update(values)
            self.path.write_text(json.dumps(self._data, indent=2))


def _ssl_context() -> ssl.SSLContext | None:
    """Verify HTTPS with certifi's CA list. Python from python.org on macOS ships without system
    certificates, so plain urllib fails there with CERTIFICATE_VERIFY_FAILED."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def _urlopen(req, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())


def _http_json(url: str, payload: dict | None = None, timeout: float = 3) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _urlopen(req, timeout) as resp:
        return json.loads(resp.read())


def _multipart(url: str, fields: dict, file_field: str, file_path: Path, timeout: float) -> dict:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
        + file_path.read_bytes() + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url, data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with _urlopen(req, timeout) as resp:
        return json.loads(resp.read())


# --------------------------------------------------------------------------- whisper

_HF = "https://huggingface.co/{}/resolve/main/{}"
_GG = "ggerganov/whisper.cpp"


def _model(file, label, size, desc, repo=_GG, path=None, pack=None, compressed_size=None):
    """`compressed_size` set = the published file is uncompressed (f16); the app compresses it to
    q5_0 with whisper-quantize right after downloading (about a third of the size, faster)."""
    return {"file": file, "label": label, "size": size, "description": desc,
            "url": _HF.format(repo, path or file), "pack": pack, "compressed_size": compressed_size}


# Every model the app can download. `pack` = language code of a language pack (None = general model).
WHISPER_CATALOG = [
    _model("ggml-tiny.bin", "Tiny", 78_000_000, "Fastest, lowest accuracy"),
    _model("ggml-base.bin", "Base", 148_000_000, "Fast, basic accuracy"),
    _model("ggml-small.bin", "Small", 488_000_000, "Good balance of speed and accuracy"),
    _model("ggml-medium.bin", "Medium", 1_534_000_000, "High accuracy, slower"),
    _model("ggml-large-v3-turbo-q5_0.bin", "Large v3 Turbo (compressed)", 574_000_000,
           "Near-best accuracy at a third of the size. Best choice for most languages other than English"),
    _model("ggml-large-v3-turbo.bin", "Large v3 Turbo", 1_625_000_000, "Near-best accuracy, fast for its size"),
    _model("ggml-large-v3.bin", "Large v3", 3_095_000_000, "Best accuracy, slowest"),

    _model("ggml-small.en.bin", "Small (English)", 488_000_000,
           "More accurate than the general Small for English", pack="en"),
    _model("ggml-medium.en-q5_0.bin", "Medium (English, compressed)", 539_000_000,
           "High English accuracy at a third of the size", pack="en"),
    _model("ggml-distil-large-v3.5.bin", "Distil-Whisper Large v3.5", 1_520_000_000,
           "Large-model English accuracy, several times faster",
           repo="distil-whisper/distil-large-v3.5-ggml", path="ggml-model.bin", pack="en"),

    _model("ggml-kb-whisper-small-q5_0.bin", "KB-Whisper Small", 175_000_000,
           "Publisher reports 7.3% word errors on Swedish vs 20.6% for general Small",
           repo="KBLab/kb-whisper-small", path="ggml-model-q5_0.bin", pack="sv"),
    _model("ggml-kb-whisper-medium-q5_0.bin", "KB-Whisper Medium", 539_000_000,
           "Publisher reports 6.6% word errors on Swedish vs 12.1% for general Medium",
           repo="KBLab/kb-whisper-medium", path="ggml-model-q5_0.bin", pack="sv"),
    _model("ggml-kb-whisper-large-q5_0.bin", "KB-Whisper Large", 1_081_000_000,
           "Publisher reports 5.4% word errors on Swedish vs 7.8% for general Large v3",
           repo="KBLab/kb-whisper-large", path="ggml-model-q5_0.bin", pack="sv"),

    _model("ggml-nb-whisper-small-q5_0.bin", "NB-Whisper Small", 175_000_000,
           "Fine-tuned Small, compressed", repo="NbAiLab/nb-whisper-small", path="ggml-model-q5_0.bin", pack="no"),
    _model("ggml-nb-whisper-medium-q5_0.bin", "NB-Whisper Medium", 539_000_000,
           "Fine-tuned Medium, compressed", repo="NbAiLab/nb-whisper-medium", path="ggml-model-q5_0.bin", pack="no"),
    _model("ggml-nb-whisper-large-q5_0.bin", "NB-Whisper Large", 1_081_000_000,
           "Fine-tuned Large, compressed", repo="NbAiLab/nb-whisper-large", path="ggml-model-q5_0.bin", pack="no"),

    _model("ggml-finnish-medium.bin", "Finnish Medium", 1_534_000_000,
           "Publisher reports 11.0% word errors on Finnish (FLEURS test)",
           repo="Finnish-NLP/Finnish-finetuned-whisper-models-ggml-format", path="ggml-model-fi-medium.bin",
           pack="fi", compressed_size=539_000_000),
    _model("ggml-finnish-large-v3.bin", "Finnish Large v3", 3_095_000_000,
           "Publisher reports 8.2% word errors on Finnish vs 12.0% for general Large v3 (FLEURS test)",
           repo="Finnish-NLP/Finnish-finetuned-whisper-models-ggml-format", path="ggml-model-fi-large-v3.bin",
           pack="fi", compressed_size=1_081_000_000),

    _model("ggml-russian-large-v3-turbo.bin", "Russian Large v3 Turbo", 1_625_000_000,
           "Large v3 Turbo fine-tuned on 118,000 Russian recordings",
           repo="dvislobokov/whisper-large-v3-turbo-russian", path="ggml-model.bin",
           pack="ru", compressed_size=574_000_000),
]
CATALOG_BY_FILE = {m["file"]: m for m in WHISPER_CATALOG}

# Language packs: models fine-tuned for one language, published in whisper.cpp format.
# "recommended": False = in our own test the general Large v3 Turbo (compressed) was more accurate,
# so the app suggests that instead; "tested" is shown to the user.
LANGUAGE_PACKS = {
    "en": {"name": "English", "languages": ["en"], "by": "OpenAI and Hugging Face (Distil-Whisper)",
           "about": "English-only models: more accurate and faster for English, but they can't transcribe other languages.",
           "link": "https://huggingface.co/ggerganov/whisper.cpp"},
    "sv": {"name": "Swedish", "languages": ["sv"], "by": "KBLab, National Library of Sweden",
           "about": "Trained on over 50,000 hours of Swedish speech. Far fewer errors on Swedish than the general models.",
           "link": "https://huggingface.co/KBLab/kb-whisper-large"},
    "no": {"name": "Norwegian", "languages": ["no", "nn"], "by": "National Library of Norway",
           "about": "Trained on Norwegian speech; writes Bokmål or Nynorsk.",
           "link": "https://huggingface.co/NbAiLab/nb-whisper-large"},
    "fi": {"name": "Finnish", "languages": ["fi"], "by": "the Finnish-NLP community",
           "about": "Fine-tuned on Finnish speech. Its authors report about a third fewer errors than the general Large v3.",
           "link": "https://huggingface.co/Finnish-NLP/whisper-large-finnish-v3",
           "recommended": False,
           "tested": "Tested on 40 real Finnish recordings (FLEURS): Finnish Medium made 11.8% word errors, "
                     "the general Large v3 Turbo (compressed) 6.6%, general Small 26.6%. Finnish Large v3 was not tested."},
    "ru": {"name": "Russian", "languages": ["ru"], "by": "dvislobokov on Hugging Face",
           "about": "Large v3 Turbo fine-tuned on Russian speech from Mozilla Common Voice 17. The author doesn't publish accuracy figures.",
           "link": "https://huggingface.co/dvislobokov/whisper-large-v3-turbo-russian",
           "recommended": False,
           "tested": "Tested on 40 real Russian recordings (FLEURS): the Russian pack made 8.4% word errors, the general "
                     "Large v3 Turbo (compressed) 4.1%, general Small 10.5%. It also writes numbers as words."},
}

# Every language Whisper understands: (code, name whisper.cpp reports). Display names are title-cased
# unless overridden below.
WHISPER_LANGUAGES = [
    ("af", "afrikaans"), ("sq", "albanian"), ("am", "amharic"), ("ar", "arabic"), ("hy", "armenian"),
    ("as", "assamese"), ("az", "azerbaijani"), ("ba", "bashkir"), ("eu", "basque"), ("be", "belarusian"),
    ("bn", "bengali"), ("bs", "bosnian"), ("br", "breton"), ("bg", "bulgarian"), ("my", "myanmar"),
    ("yue", "cantonese"), ("ca", "catalan"), ("zh", "chinese"), ("hr", "croatian"), ("cs", "czech"),
    ("da", "danish"), ("nl", "dutch"), ("en", "english"), ("et", "estonian"), ("fo", "faroese"),
    ("fi", "finnish"), ("fr", "french"), ("gl", "galician"), ("ka", "georgian"), ("de", "german"),
    ("el", "greek"), ("gu", "gujarati"), ("ht", "haitian creole"), ("ha", "hausa"), ("haw", "hawaiian"),
    ("he", "hebrew"), ("hi", "hindi"), ("hu", "hungarian"), ("is", "icelandic"), ("id", "indonesian"),
    ("it", "italian"), ("ja", "japanese"), ("jw", "javanese"), ("kn", "kannada"), ("kk", "kazakh"),
    ("km", "khmer"), ("ko", "korean"), ("lo", "lao"), ("la", "latin"), ("lv", "latvian"),
    ("ln", "lingala"), ("lt", "lithuanian"), ("lb", "luxembourgish"), ("mk", "macedonian"),
    ("mg", "malagasy"), ("ms", "malay"), ("ml", "malayalam"), ("mt", "maltese"), ("mi", "maori"),
    ("mr", "marathi"), ("mn", "mongolian"), ("ne", "nepali"), ("no", "norwegian"), ("nn", "nynorsk"),
    ("oc", "occitan"), ("ps", "pashto"), ("fa", "persian"), ("pl", "polish"), ("pt", "portuguese"),
    ("pa", "punjabi"), ("ro", "romanian"), ("ru", "russian"), ("sa", "sanskrit"), ("sr", "serbian"),
    ("sn", "shona"), ("sd", "sindhi"), ("si", "sinhala"), ("sk", "slovak"), ("sl", "slovenian"),
    ("so", "somali"), ("es", "spanish"), ("su", "sundanese"), ("sw", "swahili"), ("sv", "swedish"),
    ("tl", "tagalog"), ("tg", "tajik"), ("ta", "tamil"), ("tt", "tatar"), ("te", "telugu"),
    ("th", "thai"), ("bo", "tibetan"), ("tr", "turkish"), ("tk", "turkmen"), ("uk", "ukrainian"),
    ("ur", "urdu"), ("uz", "uzbek"), ("vi", "vietnamese"), ("cy", "welsh"), ("yi", "yiddish"),
    ("yo", "yoruba"),
]
_DISPLAY = {"my": "Burmese", "nn": "Norwegian Nynorsk", "no": "Norwegian", "ht": "Haitian Creole"}
LANGUAGE_NAMES = {code: _DISPLAY.get(code, name.title()) for code, name in WHISPER_LANGUAGES}
# whisper-server reports language names, whisper-cli reports codes; normalise to codes.
LANG_CODES = {name: code for code, name in WHISPER_LANGUAGES}


def is_english_only(model_file: str) -> bool:
    entry = CATALOG_BY_FILE.get(model_file)
    return (entry is not None and entry["pack"] == "en") or ".en" in model_file


class WhisperManager:
    def __init__(self, models_dir: Path, settings: Settings, log_path: Path):
        self.models_dir = models_dir
        self.settings = settings
        self.log_path = log_path
        self.cli = find_tool("whisper-cli", "WHISPER_BIN") or "whisper-cli"
        self.server_bin = find_tool("whisper-server", "WHISPER_SERVER_BIN")
        self.threads = os.environ.get("WHISPER_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
        self.port = int(os.environ.get("WHISPER_SERVER_PORT", "8178"))
        self._lock = threading.RLock()  # guards the server process; held during inference
        self._proc: subprocess.Popen | None = None
        self.state = "off"  # off | starting | on | error
        self.error: str | None = None
        self.loaded_model: str | None = None
        self.downloads: dict[str, dict] = {}
        models_dir.mkdir(exist_ok=True)
        atexit.register(self._stop)

    # ---- model files

    def model_path(self, name: str | None = None) -> Path:
        return self.models_dir / (name or self.settings.get("whisper_model"))

    def list_models(self) -> list[dict]:
        local = {p.name for p in self.models_dir.glob("ggml-*.bin")}
        order = [m["file"] for m in WHISPER_CATALOG] + sorted(local - set(CATALOG_BY_FILE))
        out = []
        for name in order:
            path = self.models_dir / name
            entry = CATALOG_BY_FILE.get(name) or {
                "label": name.removeprefix("ggml-").removesuffix(".bin"), "size": 0,
                "description": "Model you added to the models folder", "pack": None, "compressed_size": None,
            }
            size, dl = entry["size"], self.downloads.get(name)
            out.append({
                "name": name, "label": entry["label"], "description": entry["description"],
                "pack": entry["pack"], "english_only": is_english_only(name),
                "downloaded": path.exists(),
                "size": path.stat().st_size if path.exists() else size,
                "custom": name not in CATALOG_BY_FILE,
                "compressed_size": None if path.exists() else entry["compressed_size"],
                "selected": name == self.settings.get("whisper_model"),
                "download": {k: dl[k] for k in ("done", "total", "error", "phase")} if dl else None,
            })
        return out

    def download(self, name: str) -> None:
        if name not in CATALOG_BY_FILE:
            raise ValueError("unknown model")
        if name in self.downloads and not self.downloads[name].get("error"):
            return
        state = {"done": 0, "total": 0, "error": None, "phase": "downloading", "cancel": threading.Event()}
        self.downloads[name] = state

        def run():
            part = self.models_dir / f"{name}.part"
            try:
                with _urlopen(CATALOG_BY_FILE[name]["url"], 30) as resp:
                    state["total"] = int(resp.headers.get("Content-Length") or 0)
                    with open(part, "wb") as fh:
                        while chunk := resp.read(1 << 20):
                            if state["cancel"].is_set():
                                raise InterruptedError("cancelled")
                            fh.write(chunk)
                            state["done"] += len(chunk)
                self._finish_download(name, part, state)
                self.downloads.pop(name, None)
            except Exception as e:  # noqa: BLE001
                part.unlink(missing_ok=True)
                if isinstance(e, InterruptedError):
                    self.downloads.pop(name, None)
                else:
                    state["error"] = str(getattr(e, "reason", e))

        threading.Thread(target=run, daemon=True).start()

    def _finish_download(self, name: str, part: Path, state: dict) -> None:
        """Move a completed download into place, compressing it first if the catalog says so."""
        final = self.models_dir / name
        quantize = find_tool("whisper-quantize", "WHISPER_QUANTIZE_BIN")
        if CATALOG_BY_FILE[name]["compressed_size"] and quantize:
            state["phase"] = "compressing"
            tmp = self.models_dir / f"{name}.q5_0.part"
            proc = subprocess.run([quantize, str(part), str(tmp), "q5_0"], capture_output=True, text=True)
            if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                part.unlink()
                tmp.rename(final)
                return
            tmp.unlink(missing_ok=True)  # compression failed: keep the uncompressed model, it works too
        part.rename(final)

    def cancel_download(self, name: str) -> None:
        if name in self.downloads:
            self.downloads[name]["cancel"].set()
            if self.downloads[name].get("error"):
                self.downloads.pop(name, None)

    def delete(self, name: str) -> None:
        if name == self.settings.get("whisper_model"):
            raise ValueError("Can't delete the model that is currently selected.")
        path = self.models_dir / name
        if path.parent != self.models_dir or not name.endswith(".bin"):
            raise ValueError("bad model name")
        path.unlink(missing_ok=True)

    # ---- on / off

    def status(self) -> dict:
        alive = self._proc is not None and self._proc.poll() is None
        if self.state == "on" and not alive:
            self.state, self.error = "error", "whisper-server exited unexpectedly (see whisper-server.log)"
        return {
            "state": self.state, "error": self.error,
            "loaded_model": self.loaded_model if self.state == "on" else None,
            "selected": self.settings.get("whisper_model"),
            "language": self.settings.get("whisper_language"),
            "server_available": self.server_bin is not None,
            "cli_available": find_tool("whisper-cli", "WHISPER_BIN") is not None,
            "models": self.list_models(),
            "languages": LANGUAGE_NAMES,
            "packs": LANGUAGE_PACKS,
        }

    def set_language(self, code: str) -> None:
        if code != "auto" and code not in LANGUAGE_NAMES:
            raise ValueError(f"Unknown language: {code}")
        self.settings.update(whisper_language=code)

    def set_on(self, on: bool) -> None:
        self.settings.update(whisper_on=on)
        if on:
            self.state, self.error = "starting", None
            threading.Thread(target=self._start, daemon=True).start()
        else:
            threading.Thread(target=self._stop, daemon=True).start()

    def set_model(self, name: str) -> None:
        if not self.model_path(name).exists():
            raise ValueError("Download the model first.")
        self.settings.update(whisper_model=name)
        if self.state in ("on", "starting", "error") and self.settings.get("whisper_on"):
            self.state, self.error = "starting", None
            threading.Thread(target=self._start, daemon=True).start()

    def _start(self) -> None:
        with self._lock:
            self._stop()
            self.state, self.error = "starting", None
            path = self.model_path()
            if not self.server_bin:
                self.state, self.error = "error", "whisper-server is not installed (brew install whisper-cpp)"
                return
            if not path.exists():
                self.state, self.error = "error", f"Model {path.name} is not downloaded"
                return
            log = open(self.log_path, "ab")
            self._proc = subprocess.Popen(
                [self.server_bin, "-m", str(path), "--host", "127.0.0.1", "--port", str(self.port),
                 "-t", self.threads, "-l", "auto"],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
            deadline = time.time() + 180
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    self.state, self.error = "error", "whisper-server failed to start (see whisper-server.log)"
                    self._proc = None
                    return
                try:
                    if _http_json(f"http://127.0.0.1:{self.port}/health", timeout=1).get("status") == "ok":
                        self.state, self.loaded_model = "on", path.name
                        return
                except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                    pass
                time.sleep(0.3)
            self._stop()
            self.state, self.error = "error", "whisper-server did not become ready in time"

    def _stop(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
            self.state, self.loaded_model = "off", None

    def restore(self) -> None:
        """Turn the server back on at app start if it was on last time."""
        if self.settings.get("whisper_on") and self.model_path().exists():
            self.set_on(True)

    # ---- transcription

    def transcribe(self, wav: Path, out_base: Path) -> tuple[list[dict], str, str]:
        """Returns (segments in ms, language code, mode) where mode is 'server' or 'cli'."""
        language = self.settings.get("whisper_language") or "auto"
        model = self.settings.get("whisper_model")
        if is_english_only(model) and language not in ("auto", "en"):
            raise RuntimeError(
                f"The selected speech model only understands English, but the spoken language is set to "
                f"{LANGUAGE_NAMES.get(language, language)}. Pick a general model or a matching language pack under Models."
            )
        with self._lock:
            if self.state == "on" and self._proc and self._proc.poll() is None:
                data = _multipart(
                    f"http://127.0.0.1:{self.port}/inference",
                    {"response_format": "verbose_json", "language": language},
                    "file", wav, timeout=3600,
                )
                if "error" in data:
                    raise RuntimeError(f"whisper-server: {data['error']}")
                segments = [
                    {"start": int(s["start"] * 1000), "end": int(s["end"] * 1000), "text": s["text"].strip()}
                    for s in data.get("segments", [])
                ]
                lang = data.get("language", language)
                return _clean(segments), LANG_CODES.get(lang, lang), "server"

        path = self.model_path()
        if not path.exists():
            raise RuntimeError(f"Whisper model {path.name} is not downloaded. Pick one under Models.")
        if not find_tool("whisper-cli", "WHISPER_BIN"):
            raise RuntimeError("whisper.cpp is not installed. Run ./run.sh to set it up.")
        proc = subprocess.run(
            [self.cli, "-m", str(path), "-f", str(wav), "-l", language, "-t", self.threads,
             "-oj", "-of", str(out_base), "-np"],
            capture_output=True, text=True,
        )
        json_path = Path(f"{out_base}.json")
        if proc.returncode != 0:
            raise RuntimeError(f"whisper-cli failed: {proc.stderr.strip()[-800:]}")
        data = json.loads(json_path.read_text(errors="replace"))
        json_path.unlink(missing_ok=True)
        segments = [
            {"start": s["offsets"]["from"], "end": s["offsets"]["to"], "text": s.get("text", "").strip()}
            for s in data.get("transcription", [])
        ]
        return _clean(segments), data.get("result", {}).get("language", language), "cli"


def _clean(segments: list[dict]) -> list[dict]:
    return [s for s in segments if s["text"] and s["text"] not in ("[BLANK_AUDIO]", "[ Silence ]")]


# --------------------------------------------------------------------------- LLM


class LLMManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lms = lms_path()
        self.busy: str | None = None  # what an on/off/load operation is doing right now
        self.error: str | None = None
        self._op_lock = threading.Lock()
        self._cache: dict = {}
        self._cache_at = 0.0

    # ---- LM Studio helpers

    def _lms(self, *args, timeout=15) -> str:
        proc = subprocess.run([self.lms, *args], capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout).strip()[-500:] or f"lms {args[0]} failed")
        return proc.stdout

    def _lms_json(self, *args):
        out = self._lms(*args)  # may start with a status line such as "Waking up LM Studio service..."
        return json.loads(out[min(i for i in (out.find("{"), out.find("[")) if i >= 0):])

    def _lmstudio_status(self, fresh=False) -> dict:
        if not fresh and time.time() - self._cache_at < 1.0:
            return self._cache
        if not self.lms:
            return {"installed": False, "server_running": False, "port": None, "models": [], "loaded": []}
        try:
            server = self._lms_json("server", "status", "--json")
            loaded = [m for m in self._lms_json("ps", "--json") if m.get("type") == "llm"]
            models = [m for m in self._lms_json("ls", "--json") if m.get("type") == "llm"]
        except Exception as e:  # noqa: BLE001
            return {"installed": True, "server_running": False, "port": None, "models": [], "loaded": [],
                    "error": f"Could not talk to LM Studio: {e}"}
        loaded_by_key = {m["modelKey"]: m for m in loaded}
        self._cache = {
            "installed": True,
            "server_running": bool(server.get("running")),
            "port": server.get("port"),
            "models": [{
                "key": m["modelKey"],
                "name": m.get("displayName") or m["modelKey"],
                "params": m.get("paramsString"),
                "quant": (m.get("quantization") or {}).get("name"),
                "size": m.get("sizeBytes"),
                "max_context": m.get("maxContextLength"),
                "loaded": m["modelKey"] in loaded_by_key,
                "context": loaded_by_key.get(m["modelKey"], {}).get("contextLength"),
            } for m in models],
            "loaded": [{"key": m["modelKey"], "identifier": m.get("identifier", m["modelKey"]),
                        "context": m.get("contextLength")} for m in loaded],
        }
        self._cache_at = time.time()
        return self._cache

    # ---- custom (OpenAI-compatible) server

    def _custom_status(self) -> dict:
        url = self.settings.get("custom_base_url")
        try:
            ids = [m["id"] for m in _http_json(f"{url}/models", timeout=2).get("data", [])]
            return {"reachable": True, "models": [i for i in ids if "embed" not in i.lower()]}
        except Exception as e:  # noqa: BLE001
            return {"reachable": False, "models": [], "error": str(getattr(e, "reason", e))}

    # ---- public API

    def status(self) -> dict:
        provider = self.settings.get("llm_provider")
        out = {
            "provider": provider, "busy": self.busy, "error": self.error,
            "selected": self.settings.get("llm_model"),  # replaced below with what is available
            "context_length": self.settings.get("llm_context_length"),
            "custom_base_url": self.settings.get("custom_base_url"),
            "custom_model": self.settings.get("custom_model"),
            "summary_language": self.settings.get("summary_language"),
        }
        if provider == "lmstudio":
            lm = self._lmstudio_status()
            out["lmstudio"] = lm
            out["selected"] = self._lmstudio_model(lm)
            sel = next((m for m in lm["models"] if m["key"] == out["selected"]), None)
            out["on"] = bool(lm["server_running"] and sel and sel["loaded"])
            out["active_model"] = sel["name"] if out["on"] else None
        else:
            cs = self._custom_status()
            out["custom"] = cs
            out["on"] = cs["reachable"] and bool(cs["models"])
            out["active_model"] = self._custom_model(cs) if out["on"] else None
        return out

    def _lmstudio_model(self, lm: dict) -> str | None:
        """The chosen model if it is downloaded, else the first LLM this machine has."""
        keys = [m["key"] for m in lm["models"]]
        chosen = self.settings.get("llm_model")
        return chosen if chosen in keys else (keys[0] if keys else None)

    def _custom_model(self, cs: dict) -> str | None:
        chosen = self.settings.get("custom_model")
        return chosen if chosen in cs["models"] else (cs["models"][0] if cs["models"] else None)

    def _run(self, label: str, fn) -> None:
        if not self._op_lock.acquire(blocking=False):
            raise RuntimeError("Another model operation is still running.")
        self.busy, self.error = label, None

        def wrapper():
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
            finally:
                self._cache_at = 0
                self.busy = None
                self._op_lock.release()

        threading.Thread(target=wrapper, daemon=True).start()

    def _ensure_loaded(self) -> None:
        """Start the server, unload other LLMs to free memory, load the selected model."""
        lm = self._lmstudio_status(fresh=True)
        key = self._lmstudio_model(lm)
        if not key:
            raise RuntimeError("LM Studio has no LLM downloaded yet. Download one in the LM Studio app first.")
        ctx = int(self.settings.get("llm_context_length") or 8192)
        if not lm["server_running"]:
            self.busy = "starting server"
            self._lms("server", "start", timeout=60)
        for m in lm["loaded"]:
            if m["key"] != key or (m["context"] and m["context"] != ctx):
                self.busy = f"unloading {m['key']}"
                self._lms("unload", m["identifier"], timeout=60)
        lm = self._lmstudio_status(fresh=True)
        if not any(m["key"] == key for m in lm["loaded"]):
            max_ctx = next((m["max_context"] for m in lm["models"] if m["key"] == key), None)
            if max_ctx:
                ctx = min(ctx, max_ctx)
            self.busy = "loading model"
            self._lms("load", key, "-y", "-c", str(ctx), "--identifier", key, timeout=600)

    def set_on(self, on: bool) -> None:
        if self.settings.get("llm_provider") != "lmstudio":
            raise RuntimeError("This server is started and stopped outside the app.")
        if not self.lms:
            raise RuntimeError("LM Studio (lms) is not installed.")
        if on:
            self._run("starting", self._ensure_loaded)
        else:
            def off():
                self.busy = "unloading models"
                self._lms("unload", "--all", timeout=60)
                self.busy = "stopping server"
                self._lms("server", "stop", timeout=30)
            self._run("stopping", off)

    def set_model(self, key: str) -> None:
        if self.settings.get("llm_provider") != "lmstudio":
            self.settings.update(custom_model=key)
            return
        was_on = self.status().get("on")
        self.settings.update(llm_model=key)
        if was_on:
            self._run("switching model", self._ensure_loaded)

    def set_context_length(self, n: int) -> None:
        was_on = self.status().get("on")
        self.settings.update(llm_context_length=n)
        if was_on and self.settings.get("llm_provider") == "lmstudio":
            self._run("reloading model", self._ensure_loaded)

    # ---- used by the summarizer

    def endpoint(self) -> tuple[str, str] | None:
        """(base_url, model) when the LLM is ready to answer, else None."""
        st = self.status()
        if not st["on"]:
            return None
        if st["provider"] == "lmstudio":
            return f"http://localhost:{st['lmstudio']['port']}/v1", st["selected"]
        return self.settings.get("custom_base_url").rstrip("/"), st["active_model"]
