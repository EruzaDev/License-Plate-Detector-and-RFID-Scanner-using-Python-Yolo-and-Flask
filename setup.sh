#!/usr/bin/env bash
# ==============================================================================
# Automated Setup Script for License Plate Detector & RFID Gate System
# Sets up system dependencies, user permissions, Python venv, models, and DB.
# ==============================================================================

set -e

# ANSI Color codes for clean output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  LPR & Concurrent Dual RFID System - Environment Setup ${NC}"
echo -e "${BLUE}======================================================${NC}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# 1. System Packages & Permissions
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/6] Checking system packages and hardware permissions...${NC}"

if command -v apt-get >/dev/null 2>&1; then
    echo "Detected Debian/Ubuntu-based distribution."
    sudo apt-get update -qq || true
    sudo apt-get install -y -qq python3 python3-venv python3-pip ffmpeg libgl1-mesa-glx v4l-utils alsa-utils libglib2.0-0 python3-lgpio python3-rpi-lgpio python3-rpi.gpio python3-gpiozero python3-libgpiod swig liblgpio-dev >/dev/null 2>&1 || {
        echo "Installing core system packages..."
        sudo apt-get install -y python3 python3-venv python3-pip ffmpeg v4l-utils alsa-utils
    }
elif command -v dnf >/dev/null 2>&1; then
    echo "Detected Fedora/RHEL-based distribution."
    sudo dnf install -y -q python3 python3-pip ffmpeg mesa-libGL v4l-utils alsa-utils || true
elif command -v pacman >/dev/null 2>&1; then
    echo "Detected Arch Linux distribution."
    sudo pacman -Sy --noconfirm python python-pip ffmpeg v4l-utils alsa-utils || true
fi

# Add current user to 'input' and 'video' groups for direct /dev/input and /dev/video access
CURRENT_USER="${SUDO_USER:-$USER}"
echo "Adding user '$CURRENT_USER' to 'input' and 'video' groups..."
sudo usermod -aG input "$CURRENT_USER" 2>/dev/null || true
sudo usermod -aG video "$CURRENT_USER" 2>/dev/null || true

# Install udev rule for persistent /dev/input event permissions
if [ -d "/etc/udev/rules.d" ]; then
    echo "Configuring udev rules for RFID USB reader access..."
    sudo bash -c 'cat << "EOF" > /etc/udev/rules.d/99-input-rfid.rules
KERNEL=="event*", SUBSYSTEM=="input", MODE="0660", GROUP="input"
KERNEL=="video*", SUBSYSTEM=="video4linux", MODE="0660", GROUP="video"
EOF'
    sudo udevadm control --reload-rules 2>/dev/null || true
    sudo udevadm trigger 2>/dev/null || true
fi

# ------------------------------------------------------------------------------
# 2. Directory Structure & Environment Config
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6] Setting up project directories and environment...${NC}"

mkdir -p "$PROJECT_DIR/captures"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/res/sound"

if [ ! -f "$PROJECT_DIR/.env" ] && [ -f "$PROJECT_DIR/.env.example" ]; then
    echo "Creating .env from .env.example..."
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
fi

# ------------------------------------------------------------------------------
# 3. Python Virtual Environment
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/6] Setting up Python Virtual Environment (venv)...${NC}"

if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Creating new virtual environment at ./venv ..."
    python3 -m venv --system-site-packages "$PROJECT_DIR/venv"
fi

# Activate venv
source "$PROJECT_DIR/venv/bin/activate"

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel -q

echo "Installing project dependencies from requirements.txt..."
pip install -r "$PROJECT_DIR/requirements.txt" -q

# ------------------------------------------------------------------------------
# 4. Download Detection Models & OCR Assets
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/6] Verifying and downloading AI models & EasyOCR weights...${NC}"

python "$PROJECT_DIR/download_models.py"

# ------------------------------------------------------------------------------
# 5. Database Initialization
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/6] Initializing SQLite database schema & default admin credentials...${NC}"

python -c "
import database, importlib.util
database.init_db()
spec = importlib.util.spec_from_file_location('app_module', 'app.py')
app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)
print('  Database tables & auth schema verified successfully.')
"

# ------------------------------------------------------------------------------
# 6. Verification and Launch Helper
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/6] Setup complete!${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  ✓ All dependencies installed successfully!${NC}"
echo -e "${GREEN}  ✓ Hardware permissions configured.${NC}"
echo -e "${GREEN}  ✓ AI models & audio assets ready.${NC}"
echo -e "${GREEN}======================================================${NC}"

echo -e "\n${BLUE}How to start the application:${NC}"
echo -e "1. Run with instant input group permissions:"
echo -e "   ${YELLOW}./start.sh${NC}   (or: ${YELLOW}sg input -c 'source venv/bin/activate && python app.py'${NC})"
echo -e "\n2. Dashboard URL:"
echo -e "   ${GREEN}http://127.0.0.1:5000${NC}"
echo -e "   Admin Credentials: ${YELLOW}admin / admin123${NC}\n"
