#!/usr/bin/env bash
# Quick health check for Vive Tracker setup on Ubuntu.
# Run as your normal user.

ok() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
no() { printf "  \033[31m✗\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }

echo "=== Vive Tracker Ubuntu setup health check ==="

echo "[apt packages]"
dpkg -l steam-installer >/dev/null 2>&1 && ok "steam-installer installed" || no "steam-installer MISSING (run install_ubuntu.sh)"
dpkg -l libvulkan1 >/dev/null 2>&1 && ok "libvulkan1 installed" || no "libvulkan1 MISSING"

echo "[binaries]"
command -v steam >/dev/null && ok "steam on PATH" || no "steam NOT on PATH"
command -v vulkaninfo >/dev/null && ok "vulkaninfo on PATH" || warn "vulkaninfo missing (vulkan-tools)"

echo "[udev rules]"
if [[ -f /etc/udev/rules.d/60-HTC-Vive-perms.rules ]]; then
    ok "Vive udev rules present"
else
    no "Vive udev rules MISSING"
fi

echo "[Python deps]"
python3 -c "import openvr; print('  pyopenvr', openvr.__version__)" 2>/dev/null || no "pyopenvr MISSING (pip install --user openvr)"

echo "[USB — Valve devices visible right now]"
lsusb 2>/dev/null | grep -E "Valve|HTC|28de|0bb4" | sed 's/^/  /'
N=$(lsusb 2>/dev/null | grep -cE "Valve|HTC|28de|0bb4")
[[ $N -ge 1 ]] && ok "$N Valve/HTC USB device(s) detected" || no "No Valve/HTC USB devices — plug dongle"

echo "[Steam install location]"
for cand in "$HOME/.local/share/Steam/steamapps/common/SteamVR" "/mnt/grasp_data/SteamLibrary/steamapps/common/SteamVR" "$HOME/.steam/steam/steamapps/common/SteamVR"; do
    if [[ -d "$cand" ]]; then
        ok "SteamVR at: $cand"
        SV_FOUND=1
    fi
done
[[ -z "$SV_FOUND" ]] && no "SteamVR not installed yet (install via Steam GUI)"

echo "[null-HMD config]"
if [[ -n "$SV_FOUND" ]]; then
    NULL_FILE=$(find "$cand" -name default.vrsettings -path "*/null/*" 2>/dev/null | head -1)
    if [[ -n "$NULL_FILE" ]] && grep -q '"enable": true' "$NULL_FILE" 2>/dev/null; then
        ok "null driver enabled"
    else
        no "null driver NOT enabled (run configure_steamvr_null_hmd.sh)"
    fi
fi

echo "[ROS2 package]"
if [[ -f /home/js/franka_ros2_ws/install/vive_tracker_ros2/share/vive_tracker_ros2/package.xml ]]; then
    ok "vive_tracker_ros2 built"
else
    no "vive_tracker_ros2 NOT built (colcon build --packages-select vive_tracker_ros2)"
fi

echo ""
echo "Done."
