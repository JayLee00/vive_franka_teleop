"""
Quick CLI to list every libsurvive-tracked object (no SteamVR).
Requires PYTHONPATH + LD_LIBRARY_PATH set as in survive_tracker_node.
"""

from __future__ import annotations

import sys
import time


def main(args=None) -> None:
    try:
        import pysurvive
    except ImportError as e:
        print(f'pysurvive import failed: {e}', file=sys.stderr)
        sys.exit(1)

    try:
        ctx = pysurvive.SimpleContext(['survive_list'])
    except Exception as e:
        print(f'SimpleContext init failed: {e}. Is SteamVR closed and dongle plugged in?', file=sys.stderr)
        sys.exit(1)

    print('--- libsurvive discovered objects ---')
    for obj in ctx.Objects():
        try:    name = obj.Name().decode('utf-8', errors='replace')
        except: name = str(obj.Name())
        print(f'  {name}')

    print('\nSampling pose for 2 seconds (move trackers to verify)...')
    start = time.monotonic()
    seen: dict[str, tuple] = {}
    while ctx.Running() and time.monotonic() - start < 2.0:
        updated = ctx.NextUpdated()
        if updated is None: continue
        try:    name = updated.Name().decode('utf-8', errors='replace')
        except: name = str(updated.Name())
        p, _ts = updated.Pose()
        seen[name] = (p.Pos[0], p.Pos[1], p.Pos[2])

    print('\nLast pose per object during sample:')
    for name, (x, y, z) in seen.items():
        print(f'  {name:25s}  pos=({x:+.3f},{y:+.3f},{z:+.3f})')


if __name__ == '__main__':
    main()
