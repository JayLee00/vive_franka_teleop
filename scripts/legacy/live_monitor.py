#!/usr/bin/env python3
"""
Vive Tracker 실시간 상태 모니터 (한 화면).

사용:
    python3 /mnt/grasp_data/vive_franka_teleop/scripts/live_monitor.py

트래커를 움직이면서 valid → True 로 바뀌는 위치/각도 찾는 용도.
"""
import openvr, time, sys

CLS = {1:"HMD", 2:"Controller", 3:"Tracker", 4:"Lighthouse", 5:"Redirect"}
COLOR = {True: "\033[32m●\033[0m", False: "\033[31m✗\033[0m"}

try:
    openvr.init(openvr.VRApplication_Background)
except Exception as e:
    print(f"openvr init 실패: {e}"); sys.exit(1)

vrs = openvr.VRSystem()
print("\033[2J\033[H", end="")   # clear screen

try:
    while True:
        poses = vrs.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0.0, 8)
        rows = []
        for i in range(8):
            c = vrs.getTrackedDeviceClass(i)
            if c == 0: continue
            try:    s = vrs.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
            except: s = "?"
            p = poses[i]
            pos = ""
            if p.bPoseIsValid:
                m = p.mDeviceToAbsoluteTracking
                pos = f"({m[0][3]:+.2f},{m[1][3]:+.2f},{m[2][3]:+.2f})"
            rows.append((CLS.get(c,c), s, p.bDeviceIsConnected, p.bPoseIsValid, pos))

        print("\033[H", end="")  # cursor home
        print("─── Vive 실시간 상태 (Ctrl+C 종료) ───" + " "*40)
        print(f"{'kind':10s} {'serial':22s}  conn  valid   pos")
        for kind, s, conn, valid, pos in rows:
            print(f"{kind:10s} {s:22s}   {COLOR[bool(conn)]}     {COLOR[bool(valid)]}    {pos:30s}")
        tracker_valid = sum(1 for k,s,c,v,_ in rows if k=="Tracker" and v)
        lighthouse_valid = sum(1 for k,s,c,v,_ in rows if k=="Lighthouse" and c)
        print()
        print(f"  Lighthouses connected: {lighthouse_valid}")
        print(f"  Trackers tracking:     {tracker_valid}")
        print(f"  목표: Trackers tracking >= 1   (트래커 LED 가 녹색이면 OK)")
        sys.stdout.flush()
        time.sleep(0.3)
except KeyboardInterrupt:
    pass
finally:
    openvr.shutdown()
    print("\n종료")
