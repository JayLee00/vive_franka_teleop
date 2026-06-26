"""Launch the real Vive tracker publisher (SteamVR required)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    rate_arg = DeclareLaunchArgument('rate_hz', default_value='100.0')
    frame_arg = DeclareLaunchArgument('frame_id', default_value='vive_world')
    publish_tf_arg = DeclareLaunchArgument('publish_tf', default_value='true')
    require_arg = DeclareLaunchArgument('require_trackers', default_value='true')
    name_map_arg = DeclareLaunchArgument(
        'tracker_name_map',
        default_value='',
        description='Comma-separated "SERIAL:friendly_name" pairs (e.g. "LHR-AAA:right_hand,LHR-BBB:left_hand"). Empty -> default tracker_1, tracker_2 names.',
    )

    node = Node(
        package='vive_tracker_ros2',
        executable='tracker_node',
        name='vive_tracker_node',
        output='screen',
        parameters=[{
            'rate_hz': LaunchConfiguration('rate_hz'),
            'frame_id': LaunchConfiguration('frame_id'),
            'publish_tf': LaunchConfiguration('publish_tf'),
            'require_trackers': LaunchConfiguration('require_trackers'),
            'tracker_name_map': LaunchConfiguration('tracker_name_map'),
        }],
    )

    return LaunchDescription([
        rate_arg, frame_arg, publish_tf_arg, require_arg, name_map_arg,
        node,
    ])
