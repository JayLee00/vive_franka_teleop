#!/usr/bin/env bash
# One-shot Ubuntu setup for Vive Tracker + Lighthouse.
# Run once with sudo:   sudo bash /mnt/grasp_data/vive_franka_teleop/scripts/install_ubuntu.sh
#
# What this does (idempotent — safe to re-run):
#   1. Enable i386 architecture + multiverse (already enabled here, but checked anyway)
#   2. apt update
#   3. apt install steam-installer + dependencies for SteamVR
#   4. Install udev rules so user can access HTC/Valve USB without root
#   5. Reload udev so changes take effect immediately
#
# What this does NOT do (manual steps after):
#   - Login to Steam (GUI)
#   - Install SteamVR (Steam → Library → SteamVR → Install)
#   - Configure Library Folder = /mnt/grasp_data/SteamLibrary (Steam → Settings → Storage)
#   - Configure null-HMD (run configure_steamvr_null_hmd.sh AFTER SteamVR installed)
#   - Pair trackers (SteamVR overlay → Devices → Pair Controller)

set -e

if [[ $EUID -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES_SRC="$SCRIPT_DIR/60-HTC-Vive-perms.rules"
RULES_DST="/etc/udev/rules.d/60-HTC-Vive-perms.rules"

echo "=== [1/5] enable i386 + multiverse ==="
dpkg --add-architecture i386
apt-get update -qq

echo "=== [2/5] install steam-installer + SteamVR runtime deps ==="
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    steam-installer \
    libudev-dev \
    libsdl2-dev \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libgles2-mesa

echo "=== [3/5] install Vive/Valve udev rules ==="
if [[ ! -f "$RULES_SRC" ]]; then
    echo "ERROR: $RULES_SRC not found"; exit 1
fi
cp "$RULES_SRC" "$RULES_DST"
chmod 644 "$RULES_DST"
chown root:root "$RULES_DST"

echo "=== [4/5] reload udev ==="
udevadm control --reload-rules
udevadm trigger

echo "=== [5/5] verify ==="
which steam || echo "  (steam binary still not on PATH; first launch via 'steam' may auto-update)"
ls -la "$RULES_DST"

echo ""
echo "─── DONE. Manual steps next: ───"
echo "  1) Run as your user:  steam   (first launch downloads ~300MB Steam runtime, logs you in)"
echo "  2) Steam → Settings → Storage → '+' → /mnt/grasp_data/SteamLibrary → Make Default"
echo "  3) Steam → Library → search 'SteamVR' → Install (will land on Samsung SSD now)"
echo "  4) Quit SteamVR, then run as user:"
echo "       bash $SCRIPT_DIR/configure_steamvr_null_hmd.sh"
echo "  5) Power Lighthouse 2 base stations (set channel A/b on back), USB dongle plugged,"
echo "     tracker side-button long-press = pairing mode (blue blink),"
echo "     SteamVR → ☰ → Devices → Pair Controller → Vive Tracker → confirm green hexagon."
