# vive_3d_viz

Vive Tracker 3.0 + Lighthouse base stations 3D visualization (RViz2).
SteamVR/OpenVR backend via vendored `triad_openvr`.

## Build

```bash
ln -sf /home/js/data/Vive_Franka_tele/vive_3d_viz ~/franka_ros2_ws/src/vive_3d_viz
cd ~/franka_ros2_ws
colcon build --packages-select vive_3d_viz --symlink-install
source install/setup.bash
```

## Run

1. Start SteamVR (so trackers + lighthouses are paired/visible).
2. Confirm devices:

```bash
ros2 run vive_3d_viz list_devices
```

3. Launch viz + RViz2:

```bash
ros2 launch vive_3d_viz viz.launch.py
```

Optional: friendly names

```bash
ros2 launch vive_3d_viz viz.launch.py tracker_name_map:="LHR-9AB9C3A4:right_hand,LHR-46A73216:left_hand"
```

## Topics

- `/vive/<friendly>/pose`  geometry_msgs/PoseStamped
- `/vive/<friendly>/valid` std_msgs/Bool
- `/vive/markers`          visualization_msgs/MarkerArray
- TF: `vive_world` → `vive_<friendly>`

## Frame

SteamVR world frame: right-handed, +Y up. RViz config uses `vive_world` as Fixed Frame.
