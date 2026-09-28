#!/usr/bin/env bash
# Voice Notes: set up whatever is missing, then start the app. Works on macOS and Linux.
#
#   ./run.sh           check dependencies, offer to install missing ones, start the app
#   ./run.sh --yes     install missing dependencies without asking (for scripts)
#   ./run.sh --check   only report what is installed; change nothing
#
# For more info check README
# Environment: PORT (default 5050), NO_BROWSER=1, WHISPER_CPP_VERSION, WHISPER_CMAKE_FLAGS.
set -euo pipefail
cd "$(dirname "$0")"

WHISPER_CPP_VERSION="${WHISPER_CPP_VERSION:-1.9.4}"
PORT="${PORT:-5050}"
DEFAULT_MODEL="ggml-small.bin"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$DEFAULT_MODEL"
VENDOR="vendor/whisper.cpp"  # where whisper.cpp is built when it isn't installed system-wide
ASSUME_YES=0
CHECK_ONLY=0
MISSING=0

for arg in "$@"; do
  case "$arg" in
    -y|--yes) ASSUME_YES=1 ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (see ./run.sh --help)"; exit 2 ;;
  esac
done

#  helpers

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=; G=; Y=; R=; N=; fi
step() { printf '\n%s==>%s %s\n' "$B" "$N" "$*"; }
ok()   { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '\n%sError:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
cpu_count() { nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; }

# ask "question": returns 0 for yes. Without a terminal the answer is no, unless --yes was given.
ask() {
  [ "$ASSUME_YES" = 1 ] && return 0
  if [ ! -t 0 ]; then
    warn "$1 (no terminal to ask, so skipping; run ./run.sh --yes to accept)"
    return 1
  fi
  local reply
  read -r -p "  $1 [Y/n] " reply || return 1
  case "$reply" in [nN]*) return 1 ;; *) return 0 ;; esac
}

# Run a command as root (via sudo when needed), printing it first.
run_root() {
  if [ "$(id -u)" -ne 0 ]; then
    have sudo || die "Administrator rights are needed to run: $*"
    printf '  $ sudo %s\n' "$*"
    sudo "$@"
  else
    printf '  $ %s\n' "$*"
    "$@"
  fi
}

# download URL FILE: via a .part file, so an interrupted download never looks complete.
download() {
  local url="$1" out="$2"
  if have curl; then
    curl -fL --progress-bar -o "$out.part" "$url"
  elif have wget; then
    wget -q --show-progress -O "$out.part" "$url"
  elif [ -x .venv/bin/python ]; then
    .venv/bin/python -c 'import shutil, ssl, sys, urllib.request, certifi
ctx = ssl.create_default_context(cafile=certifi.where())
with urllib.request.urlopen(sys.argv[1], context=ctx) as r, open(sys.argv[2], "wb") as f: shutil.copyfileobj(r, f)' "$url" "$out.part"
  else
    die "Need curl or wget to download $url"
  fi || { rm -f "$out.part"; die "Download failed: $url"; }
  mv "$out.part" "$out"
}

OS="$(uname -s)"
case "$OS" in
  Darwin) OS=macos ;;
  Linux) OS=linux ;;
  *) die "Unsupported system: $OS. Voice Notes runs on macOS and Linux (on Windows, use WSL)." ;;
esac
ARCH="$(uname -m)"

PKG=""  # Linux package manager
if [ "$OS" = linux ]; then
  for p in apt-get dnf pacman zypper apk; do
    if have "$p"; then PKG="$p"; break; fi
  done
fi

# pkg_install <apt> <dnf> <pacman> <zypper> <apk>: package names for each distro family.
# shellcheck disable=SC2086  # each argument is a space-separated list of packages
pkg_install() {
  case "$PKG" in
    apt-get) run_root apt-get update -qq && run_root apt-get install -y $1 ;;
    dnf)     run_root dnf install -y $2 ;;
    pacman)  run_root pacman -S --needed --noconfirm $3 ;;
    zypper)  run_root zypper --non-interactive install $4 ;;
    apk)     run_root apk add $5 ;;
    *)       return 1 ;;
  esac
}

# --------------------------------------------------------------------------- Python

find_python() {
  local c
  for c in python3 python3.13 python3.12 python3.11 python3.10 python3.9 python; do
    if have "$c" && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
      command -v "$c"
      return 0
    fi
  done
  return 1
}

install_python() {
  if [ "$OS" = macos ]; then
    die "Python 3.9 or newer is needed. Install Apple's Command Line Tools (xcode-select --install) or Python from python.org, then run ./run.sh again."
  fi
  warn "Python 3.9 or newer is needed."
  if [ -n "$PKG" ] && ask "Install Python with $PKG (needs your admin password)?"; then
    pkg_install "python3 python3-venv" "python3" "python" "python3" "python3" || die "Installing Python failed."
  else
    die "Install Python 3.9 or newer with your package manager, then run ./run.sh again."
  fi
}

setup_venv() {
  step "Python environment"
  if [ -x .venv/bin/python ] && .venv/bin/python -c 'import flask' >/dev/null 2>&1 \
     && cmp -s requirements.txt .venv/.requirements; then
    ok "ready (Python $(.venv/bin/python -c 'import platform; print(platform.python_version())'))"
    return
  fi
  if [ "$CHECK_ONLY" = 1 ]; then warn "not set up yet"; MISSING=1; return; fi

  local py
  py="$(find_python)" || { install_python; py="$(find_python)" || die "Python 3.9+ still not found."; }

  if [ -d .venv ] && ! .venv/bin/python -c 'import sys' >/dev/null 2>&1; then
    warn "The existing .venv doesn't work here (probably copied from another computer); recreating it."
    rm -rf .venv
  fi
  if [ ! -x .venv/bin/python ]; then
    if ! "$py" -m venv .venv >/dev/null 2>&1; then
      rm -rf .venv
      # Debian and Ubuntu ship the venv module as a separate package.
      [ "$PKG" = apt-get ] || die "Could not create a virtual environment with: $py -m venv .venv"
      local ver
      ver="$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
      warn "Python's venv module is missing."
      ask "Install python$ver-venv with apt (needs your admin password)?" \
        || die "Install it with: sudo apt-get install python3-venv"
      run_root apt-get install -y "python$ver-venv" || run_root apt-get install -y python3-venv
      "$py" -m venv .venv || die "Could not create the virtual environment."
    fi
  fi
  echo "  Installing Python packages…"
  .venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt \
    || die "Installing Python packages failed (see the messages above)."
  cp requirements.txt .venv/.requirements
  ok "ready (Python $(.venv/bin/python -c 'import platform; print(platform.python_version())'))"
}

#  ffmpeg

setup_ffmpeg() {
  step "ffmpeg (reads your audio files)"
  if have ffmpeg; then ok "$(command -v ffmpeg)"; return; fi
  if [ -x .venv/bin/python ] && .venv/bin/python -c 'import imageio_ffmpeg' >/dev/null 2>&1; then
    ok "private copy in .venv"
    return
  fi
  if [ "$CHECK_ONLY" = 1 ]; then warn "missing"; MISSING=1; return; fi

  # A static ffmpeg build from PyPI (~30 MB) that lives inside .venv: no admin rights, nothing
  # installed system-wide, so no need to ask. Available for macOS and Linux on x86_64 and arm64.
  echo "  Installing a private copy of ffmpeg into .venv…"
  if .venv/bin/python -m pip install -q --disable-pip-version-check imageio-ffmpeg \
     && .venv/bin/python -c 'import imageio_ffmpeg; imageio_ffmpeg.get_ffmpeg_exe()' >/dev/null 2>&1; then
    ok "private copy installed in .venv"
    return
  fi
  warn "No ready-made ffmpeg for this system."
  if [ -n "$PKG" ] && ask "Install ffmpeg with $PKG instead (needs your admin password)?"; then
    pkg_install ffmpeg "ffmpeg-free" ffmpeg ffmpeg ffmpeg && have ffmpeg && { ok "installed"; return; }
  fi
  die "ffmpeg is required. Install it (e.g. 'sudo apt-get install ffmpeg' or 'brew install ffmpeg') and run ./run.sh again."
}

#  whisper.cpp

ensure_build_tools() {
  if { have c++ || have g++ || have clang++; } && have make; then return; fi
  if [ "$OS" = macos ]; then
    die "Building whisper.cpp needs Apple's Command Line Tools. Run: xcode-select --install, then ./run.sh again."
  fi
  warn "Building whisper.cpp needs a C++ compiler and make."
  if [ -z "$PKG" ] || ! ask "Install build tools with $PKG (needs your admin password)?"; then
    die "Install a C++ compiler and make (e.g. 'sudo apt-get install build-essential'), then run ./run.sh again."
  fi
  pkg_install build-essential "gcc-c++ make" base-devel "gcc-c++ make" build-base \
    || die "Installing build tools failed."
}

build_whisper() {
  ensure_build_tools
  local cmake log
  cmake="$(command -v cmake || true)"
  if [ -z "$cmake" ]; then
    echo "  Installing cmake into .venv (no admin rights needed)…"
    .venv/bin/python -m pip install -q --disable-pip-version-check cmake || die "Could not install cmake."
    cmake="$PWD/.venv/bin/cmake"
  fi

  echo "  Downloading whisper.cpp $WHISPER_CPP_VERSION…"
  mkdir -p vendor
  rm -rf "$VENDOR"
  download "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v$WHISPER_CPP_VERSION.tar.gz" \
           "vendor/whisper.cpp.tar.gz"
  mkdir -p "$VENDOR"
  tar -xzf vendor/whisper.cpp.tar.gz -C "$VENDOR" --strip-components=1
  rm -f vendor/whisper.cpp.tar.gz

  # Static binaries keep working if the project folder is moved.
  local flags="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_SDL2=OFF"
  if [ "$OS" = linux ] && have nvcc; then
    echo "  NVIDIA CUDA found: building with GPU support."
    flags="$flags -DGGML_CUDA=ON"
  fi
  flags="$flags ${WHISPER_CMAKE_FLAGS:-}"

  log="$VENDOR/build.log"
  echo "  Building (this takes a few minutes; log: $log)…"
  # shellcheck disable=SC2086  # $flags is a list of separate arguments
  "$cmake" -S "$VENDOR" -B "$VENDOR/build" $flags >"$log" 2>&1 \
    || { tail -n 20 "$log"; die "Configuring whisper.cpp failed. Full log: $log"; }
  "$cmake" --build "$VENDOR/build" --config Release -j "$(cpu_count)" --target whisper-cli whisper-server whisper-quantize >>"$log" 2>&1 \
    || { tail -n 30 "$log"; die "Building whisper.cpp failed. Full log: $log"; }
  [ -x "$VENDOR/build/bin/whisper-cli" ] || die "The build finished but whisper-cli is missing. Log: $log"
  ok "built into $VENDOR/build/bin"
}

setup_whisper() {
  step "whisper.cpp (speech recognition)"
  local cli="" server=""
  if have whisper-cli; then
    cli="$(command -v whisper-cli)"
    server="$(command -v whisper-server || true)"
  elif [ -x "$VENDOR/build/bin/whisper-cli" ]; then
    cli="$VENDOR/build/bin/whisper-cli"
    [ -x "$VENDOR/build/bin/whisper-server" ] && server="$VENDOR/build/bin/whisper-server"
  fi
  if [ -n "$cli" ]; then
    ok "$cli"
    [ -n "$server" ] || warn "whisper-server not found, so Whisper can't stay loaded in memory (transcription still works)."
    return
  fi
  if [ "$CHECK_ONLY" = 1 ]; then warn "missing"; MISSING=1; return; fi

  if have brew && ask "whisper.cpp is missing. Install it with Homebrew (ready-made)?"; then
    brew install whisper-cpp && have whisper-cli && { ok "installed"; return; }
    warn "Homebrew install failed; building from source instead."
  fi
  ask "Build whisper.cpp $WHISPER_CPP_VERSION from source into $VENDOR (a few minutes)?" \
    || die "whisper.cpp is required. Install it (e.g. 'brew install whisper-cpp') and run ./run.sh again."
  build_whisper
}

#  models & LLM

setup_model() {
  step "Speech model"
  local count
  count="$(find models -maxdepth 1 -name 'ggml-*.bin' 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$count" -gt 0 ]; then ok "$count model(s) in models/"; return; fi
  if [ -n "${WHISPER_MODEL:-}" ]; then ok "using WHISPER_MODEL=$WHISPER_MODEL"; return; fi
  if [ "$CHECK_ONLY" = 1 ]; then warn "none downloaded yet"; MISSING=1; return; fi
  if ask "Download the default speech model (Small, 488 MB, all languages)? Others can be added in the app."; then
    mkdir -p models
    download "$MODEL_URL" "models/$DEFAULT_MODEL"
    ok "models/$DEFAULT_MODEL"
  else
    warn "No speech model yet: download one in the app (Models panel) before transcribing."
  fi
}

check_llm() {
  step "Local LLM for key moments (optional)"
  if have lms || [ -x "$HOME/.lmstudio/bin/lms" ]; then
    ok "LM Studio found"
  elif have curl && curl -sf -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama is running (choose \"Other server\" in the app's Models panel)"
  elif have ollama; then
    ok "Ollama installed (start it with 'ollama serve'; choose \"Other server\" in the Models panel)"
  else
    warn "None found. Transcription works without one; for key moments install LM Studio (lmstudio.ai) or Ollama (ollama.com)."
  fi
}

port_busy() {
  .venv/bin/python -c 'import socket, sys; s = socket.socket(); sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "$PORT"
}

open_browser() {
  [ "${NO_BROWSER:-0}" = 1 ] && return 0
  local url="http://localhost:$PORT"
  if [ "$OS" = macos ]; then
    (sleep 1.5; open "$url") >/dev/null 2>&1 &
  elif have xdg-open && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    (sleep 1.5; xdg-open "$url") >/dev/null 2>&1 &
  fi
  return 0
}

#  main

# When sourced (e.g. by tests), only define the functions above.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then return 0; fi

printf '%sVoice Notes%s · %s %s\n' "$B" "$N" "$OS" "$ARCH"
setup_venv
setup_ffmpeg
setup_whisper
setup_model
check_llm

if [ "$CHECK_ONLY" = 1 ]; then
  echo
  if [ "$MISSING" = 1 ]; then echo "Some parts are missing. Run ./run.sh to set them up."; exit 1; fi
  echo "Everything is set up."
  exit 0
fi

if port_busy; then
  die "Port $PORT is already in use. Is Voice Notes already running? To use another port: PORT=5051 ./run.sh"
fi
step "Starting Voice Notes at ${B}http://localhost:$PORT${N}  (Ctrl+C to stop)"
open_browser
export PORT
exec .venv/bin/python app.py
