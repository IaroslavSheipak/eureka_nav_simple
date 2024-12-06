from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

navcam_dir = get_package_share_directory('eureka_camera_2')

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            launch_description_source = PythonLaunchDescriptionSource(navcam_dir + '/launch/eureka_navcam_simple_launch.py'),
        ),
 #       Node(
 #           package='eureka_nav_simple',
  #          executable='nav_simple',
  #          name='nav_simple',
 #           shell=True,
  #      ),
        Node(
            package='eureka_nav_simple',
            executable='nav_simple_cv',
            name='nav_simple_cv',
            shell=True,
        ),
    ])