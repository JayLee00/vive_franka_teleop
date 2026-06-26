"""Launch the libsurvive-based tracker publisher.

NOTE: SteamVR must NOT be running. PYTHONPATH and LD_LIBRARY_PATH must include
the vendored libsurvive — use scripts/run_survive.sh or set them in your shell.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('rate_hz', default_value='100.0'),
        DeclareLaunchArgument('frame_id', default_value='survive_world'),
        DeclareLaunchArgument('publish_tf', default_value='true'),
        DeclareLaunchArgument('tracker_name_map', default_value=''),
        DeclareLaunchArgument('survive_argv', default_value=''),
        Node(
            package='vive_teleop_3d',
            executable='survive_tracker_node',
            name='survive_tracker_node',
            output='screen',
            parameters=[{
                'rate_hz': LaunchConfiguration('rate_hz'),
                'frame_id': LaunchConfiguration('frame_id'),
                'publish_tf': LaunchConfiguration('publish_tf'),
                'tracker_name_map': LaunchConfiguration('tracker_name_map'),
                'survive_argv': LaunchConfiguration('survive_argv'),
            }],
        ),
    ])
