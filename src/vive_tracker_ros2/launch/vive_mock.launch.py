"""Launch the mock tracker (no SteamVR / hardware required)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('rate_hz', default_value='100.0'),
        DeclareLaunchArgument('tracker_name', default_value='mock_tracker'),
        Node(
            package='vive_tracker_ros2',
            executable='mock_tracker_node',
            name='vive_mock_tracker_node',
            output='screen',
            parameters=[{
                'rate_hz': LaunchConfiguration('rate_hz'),
                'tracker_name': LaunchConfiguration('tracker_name'),
            }],
        ),
    ])
