#!/usr/bin/env bash
# ==============================================================================
# Start Script for License Plate Detector & Dual RFID System
# Automatically activates venv, ensures input/video group permissions, and runs app.
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Virtual environment not found. Running setup.sh first..."
    bash "$PROJECT_DIR/setup.sh"
fi

# Ensure user is in 'input' group
if ! id -nG "$USER" | grep -qw "input"; then
    echo "Adding $USER to 'input' group for hardware USB RFID access..."
    sudo usermod -aG input,video "$USER" 2>/dev/null || true
fi

echo "Starting License Plate Detector & Dual RFID Scanner..."

# Run inside 'input' group session so /dev/input permissions are active immediately
if command -v sg >/dev/null 2>&1; then
    sg input -c "source '$PROJECT_DIR/venv/bin/activate' && python '$PROJECT_DIR/app.py'"
else
    source "$PROJECT_DIR/venv/bin/activate"
    python "$PROJECT_DIR/app.py"
fi
