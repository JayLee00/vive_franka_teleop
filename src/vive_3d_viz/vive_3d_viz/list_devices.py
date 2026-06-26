"""List all SteamVR-tracked devices (trackers + lighthouses + HMD + controllers).

Run with SteamVR running. Useful for confirming pairing and filling tracker_name_map.
"""

from __future__ import annotations

import sys


def main(args=None) -> None:
    try:
        from . import _triad_openvr as triad_openvr
    except ImportError as e:
        print(f'Failed to import triad_openvr: {e}', file=sys.stderr)
        sys.exit(1)
    try:
        vr = triad_openvr.triad_openvr()
    except Exception as e:
        print(f'triad_openvr init failed ({e}). Is SteamVR running?', file=sys.stderr)
        sys.exit(1)

    print('--- Discovered SteamVR devices ---')
    for name, dev in vr.devices.items():
        try:
            serial = dev.get_serial()
        except Exception:
            serial = '?'
        try:
            model = dev.get_model()
        except Exception:
            model = '?'
        print(f'  {name:25s}  serial={serial:20s}  model={model}')


if __name__ == '__main__':
    main()
