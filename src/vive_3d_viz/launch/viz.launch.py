"""Launch the Vive 3D viz node + RViz2 with a preset config."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('vive_3d_viz')
    rviz_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([pkg, 'rviz', 'viz.rviz']),
        description='Path to RViz2 config file.',
    )
    name_map_arg = DeclareLaunchArgument(
        'tracker_name_map',
        # R/L 고정: 시리얼 기반 매핑 (인덱스 tracker_1/2 로 떨어져 좌/우가 뒤바뀌는 것 방지)
        default_value='LHR-7B9A3BA9:right,LHR-F4A94AD1:left',
        description='Comma-separated "SERIAL:friendly_name" mappings. Default pins R/L by serial.',
    )
    rate_arg = DeclareLaunchArgument('rate_hz', default_value='60.0')

    viz_node = Node(
        package='vive_3d_viz',
        executable='viz_node',
        name='vive_3d_viz_node',
        output='screen',
        parameters=[{
            'rate_hz': LaunchConfiguration('rate_hz'),
            'tracker_name_map': LaunchConfiguration('tracker_name_map'),
            'frame_id': 'vive_world',
            'publish_tf': True,
            'require_devices': False,
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', LaunchConfiguration('rviz_config')],
    )

    return LaunchDescription([rviz_arg, name_map_arg, rate_arg, viz_node, rviz])
