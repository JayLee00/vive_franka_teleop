from setuptools import setup
from glob import glob

package_name = 'vive_tracker_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jay Lee',
    maintainer_email='jay.lee@kist.re.kr',
    description='HTC Vive Tracker pose publisher for ROS2 (Humble).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tracker_node = vive_tracker_ros2.tracker_node:main',
            'mock_tracker_node = vive_tracker_ros2.mock_tracker_node:main',
            'list_devices = vive_tracker_ros2.list_devices:main',
        ],
    },
)
