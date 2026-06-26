"""
Standalone helper: lists every SteamVR-tracked device with its serial.

Run with SteamVR already running. Useful for filling in tracker_name_map
(serial → friendly name) without spinning up the full ROS2 node.
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
        print(f'  {name:15s}  serial={serial}  model={model}')

    print()
    print('Example tracker_name_map entries (paste into launch args or YAML):')
    for name, dev in vr.devices.items():
        if name.startswith('tracker_'):
            try:
                serial = dev.get_serial()
            except Exception:
                continue
            print(f'  - "{serial}:right_hand"')


if __name__ == '__main__':
    main()
