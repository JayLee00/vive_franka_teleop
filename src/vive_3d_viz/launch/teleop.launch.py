"""Launch the Vive -> 6DoF teleop delta node (publishes JSON deltas over ROS2)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arms = DeclareLaunchArgument('arms', default_value='left,right')
    rate = DeclareLaunchArgument('rate_hz', default_value='100.0')
    pos_scale = DeclareLaunchArgument('pos_scale', default_value='1.0')
    auto_engage = DeclareLaunchArgument('auto_engage', default_value='false')

    node = Node(
        package='vive_3d_viz',
        executable='teleop_delta',
        name='vive_teleop_delta',
        output='screen',
        parameters=[{
            'arms': LaunchConfiguration('arms'),
            'rate_hz': LaunchConfiguration('rate_hz'),
            'pos_scale': LaunchConfiguration('pos_scale'),
            'auto_engage': LaunchConfiguration('auto_engage'),
        }],
    )
    return LaunchDescription([arms, rate, pos_scale, auto_engage, node])
