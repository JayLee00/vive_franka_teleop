from setuptools import setup
from glob import glob

package_name = 'vive_teleop_3d'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jay Lee',
    maintainer_email='jay.lee@kist.re.kr',
    description='HTC Vive Tracker pose publisher via libsurvive (no SteamVR).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'survive_tracker_node = vive_teleop_3d.survive_tracker_node:main',
            'survive_list = vive_teleop_3d.survive_list:main',
        ],
    },
)
