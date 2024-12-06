from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'eureka_nav_simple'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, package_name, 'csv'), glob(os.path.join(package_name,'csv', '*.csv'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eurekanuc',
    maintainer_email='andrey040902@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nav_simple = eureka_nav_simple.nav_simple:main',
            'nav_simple_cv = eureka_nav_simple.nav_simple_cv:main',
        ],
    },
)
