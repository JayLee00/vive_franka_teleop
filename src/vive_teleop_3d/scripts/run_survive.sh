#!/usr/bin/env bash
# Helper that sets PYTHONPATH + LD_LIBRARY_PATH for vendored libsurvive,
# kills SteamVR (libsurvive needs exclusive USB), and launches the ROS2 node.
#
# Usage:
#   bash /mnt/grasp_data/vive_franka_teleop/vive_teleop_3d/scripts/run_survive.sh

set -e

SURV=/home/js/Desktop/vive_franka_teleop/vendor/libsurvive

# 1. SteamVR 종료 (libsurvive 가 USB 잡으려면 SteamVR 가 떠 있으면 안 됨)
for p in vrmonitor vrserver vrcompositor vrwebhelper vrdashboard; do
    pkill "$p" 2>/dev/null || true
done
sleep 2
for p in vrmonitor vrserver vrcompositor vrwebhelper vrdashboard; do
    pkill -9 "$p" 2>/dev/null || true
done
sleep 1

# 2. ROS / libsurvive 환경
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)
unset PYTHONPATH || true
unset CONDA_PREFIX || true
export PATH
export PYTHONPATH=$SURV/bindings/python
export LD_LIBRARY_PATH=$SURV/bin

source /opt/ros/humble/setup.bash
source /home/js/franka_ros2_ws/install/setup.bash
# 위에서 setup.bash 가 PYTHONPATH 를 덮어쓸 수 있으므로 다시 prepend
export PYTHONPATH=$SURV/bindings/python:$PYTHONPATH

export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0

echo "─── env ready ───"
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

exec ros2 launch vive_teleop_3d survive_tracker.launch.py "$@"
