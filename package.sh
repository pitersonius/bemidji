#!/usr/bin/env bash
# Create a clean archive of the app to share: dist/voice-notes-<date>.tar.gz
# It contains only the program, not your recordings, models, settings or .venv.
# On the other computer:  tar -xzf voice-notes-*.tar.gz && cd voice-notes && ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

name="voice-notes"
archive="dist/$name-$(date +%Y%m%d).tar.gz"
files="app.py models.py requirements.txt run.sh package.sh README.md .gitignore static"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/$name" dist
# shellcheck disable=SC2086  # $files is a list
cp -R $files "$tmp/$name/"
find "$tmp/$name" \( -name '.DS_Store' -o -name '__pycache__' \) -prune -exec rm -rf {} +
chmod +x "$tmp/$name/run.sh" "$tmp/$name/package.sh"

# COPYFILE_DISABLE stops macOS tar from adding "._*" metadata files that show up on Linux.
COPYFILE_DISABLE=1 tar -czf "$archive" -C "$tmp" "$name"
echo "Created $archive ($(du -h "$archive" | cut -f1))"
