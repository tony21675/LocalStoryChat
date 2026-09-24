#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Use the CPU performance profile for local LLM inference.
if command -v powerprofilesctl >/dev/null 2>&1; then
    powerprofilesctl set performance || true
fi

python3 server.py
