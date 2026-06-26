from setuptools import setup
from glob import glob

package_name = 'vive_3d_viz'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jay Lee',
    maintainer_email='jay.lee@kist.re.kr',
    description='3D visualization of Vive Tracker 3.0 + Lighthouse base stations in RViz2.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'viz_node = vive_3d_viz.viz_node:main',
            'list_devices = vive_3d_viz.list_devices:main',
            'teleop_delta = vive_3d_viz.teleop_delta_node:main',
            'relative_pose = vive_3d_viz.relative_pose_node:main',
        ],
    },
)
