#!/usr/bin/env bash
# Configure SteamVR to run WITHOUT a physical HMD (Vive Tracker-only use case).
# Run AFTER SteamVR is installed (i.e., after first Steam launch + SteamVR install).
# Must run as the SteamVR user (NOT root, NOT sudo).

set -e

if [[ $EUID -eq 0 ]]; then
    echo "ERROR: run this as your normal user, not root. SteamVR config lives in your home dir."
    exit 1
fi

# Locate the Steam install. Default with steam-installer: ~/.local/share/Steam.
# But user may have set Library = /mnt/grasp_data/SteamLibrary.
SV_DEFAULT="$HOME/.local/share/Steam/steamapps/common/SteamVR"
SV_SAMSUNG="/mnt/grasp_data/SteamLibrary/steamapps/common/SteamVR"
SV_OLD="$HOME/.steam/steam/steamapps/common/SteamVR"
SV_DIR=""
for cand in "$SV_DEFAULT" "$SV_SAMSUNG" "$SV_OLD"; do
    if [[ -d "$cand" ]]; then
        SV_DIR="$cand"
        break
    fi
done

if [[ -z "$SV_DIR" ]]; then
    echo "ERROR: SteamVR install not found. Install it from Steam first."
    echo "Checked: $SV_DEFAULT, $SV_SAMSUNG, $SV_OLD"
    exit 1
fi
echo "Found SteamVR: $SV_DIR"

CFG_DIR="$HOME/.steam/steam/config"
mkdir -p "$CFG_DIR"
SV_GLOBAL="$CFG_DIR/steamvr.vrsettings"
NULL_SETTINGS="$SV_DIR/drivers/null/resources/settings/default.vrsettings"

echo ""
echo "=== [1/2] enable null driver in $NULL_SETTINGS ==="
if [[ ! -f "$NULL_SETTINGS" ]]; then
    echo "ERROR: null driver settings file missing. SteamVR install incomplete?"
    exit 1
fi
cp -n "$NULL_SETTINGS" "${NULL_SETTINGS}.bak.$(date +%s)" 2>/dev/null || true
python3 - "$NULL_SETTINGS" <<'PY'
import json, sys
p = sys.argv[1]
with open(p) as f:
    data = json.load(f)
data.setdefault('driver_null', {})
data['driver_null']['enable'] = True
data['driver_null'].setdefault('serialNumber', 'Null Serial Number')
data['driver_null'].setdefault('modelNumber', 'Null Model Number')
with open(p, 'w') as f:
    json.dump(data, f, indent=3)
print('null driver enabled')
PY

echo ""
echo "=== [2/2] set requireHmd=false in $SV_GLOBAL ==="
if [[ -f "$SV_GLOBAL" ]]; then
    cp "$SV_GLOBAL" "${SV_GLOBAL}.bak.$(date +%s)"
fi
python3 - "$SV_GLOBAL" <<'PY'
import json, os, sys
p = sys.argv[1]
data = {}
if os.path.exists(p):
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        data = {}
data.setdefault('steamvr', {})
data['steamvr']['requireHmd'] = False
data['steamvr']['forcedDriver'] = 'null'
data['steamvr']['activateMultipleDrivers'] = True
with open(p, 'w') as f:
    json.dump(data, f, indent=3)
print('global steamvr.vrsettings updated')
PY

echo ""
echo "DONE. Next:"
echo "  - Launch SteamVR (Steam → Library → SteamVR → Play, or via Vulkan launch option)"
echo "  - SteamVR should now start without an HMD, showing only Lighthouse + Tracker icons."
echo "  - Pair tracker via SteamVR overlay (Devices → Pair Controller → Vive Tracker)"
