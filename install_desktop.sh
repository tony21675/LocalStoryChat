#!/usr/bin/env bash
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_DIR="$HOME/Desktop"
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_DIR/Local Story Chat.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Local Story Chat
Comment=Start the local Dolphin story chat in your browser
Path=$APP_DIR
Exec=$APP_DIR/start.sh
Icon=utilities-terminal
Terminal=true
Categories=Utility;Development;
StartupNotify=true
EOF
chmod +x "$DESKTOP_DIR/Local Story Chat.desktop"
if command -v gio >/dev/null 2>&1; then
  gio set "$DESKTOP_DIR/Local Story Chat.desktop" metadata::trusted true 2>/dev/null || true
fi
echo "Desktop launcher installed at: $DESKTOP_DIR/Local Story Chat.desktop"
