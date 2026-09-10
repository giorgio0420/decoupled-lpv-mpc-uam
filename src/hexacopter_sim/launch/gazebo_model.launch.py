import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    robotXacroName = 'hexacopter_robot'
    namePackage = 'hexacopter_sim'
    modelFileRelativePath = 'model/robot.xacro'
    
    pathModelFile = os.path.join(get_package_share_directory(namePackage), modelFileRelativePath)
    robotDescription = xacro.process_file(pathModelFile).toxml()
    
    gazebo_rosPackageLaunch = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    )
    # Headless was tried here and made things worse, which is worth recording so
    # it is not tried again. The reasoning was sound -- the GUI runs on software
    # OpenGL and the simulated clock jumps 450 to 960 ms in a single tick, and
    # during a jump that size the arm keeps its last torque, swings to its stops,
    # and the impact throws the vehicle 0.7 rad. But '-s' loads fast enough that
    # the world had advanced five metres of free fall before the controller saw
    # its first sample, and the stalls did not go away with it: one per 32 s
    # either way, with the attitude excursion three times larger. The loop
    # recovers from a stall on its own, so this is a nuisance, not the blocker.
    # No '-r'. The world must stay paused until the controller has closed the
    # loop, which is what the run script does: start paused, engage, then release.
    #
    # With '-r' the world ran from launch, and the six seconds the launch takes are
    # six seconds of free fall. The vehicle engaged at z = +0.990 -- on the ground,
    # with the arm resting on the floor and curled to [0, -pi, 3.03]. From there no
    # thrust recovers it: the joints are on a contact, so the reaction terms read a
    # pinned plant and the rotors are fighting the floor. Every "it will not hold
    # attitude" run was starting from that.
    #
    # Spawning higher does not fix it and cannot: six seconds of free fall is 176
    # metres. The pause is the fix; the height only sets the margin afterwards.
    # UAM_GUI=0 starts the server alone. The window is not lost by doing so: the
    # server and the GUI have always been two processes, and the GUI can be
    # attached to a running server at any moment, including mid-run, with
    #
    #     ign gazebo -g
    #
    # That is worth having because the GUI is expensive here. WSL has no working
    # hardware path for it -- the d3d12 GL translation raises
    # Ogre::UnimplementedException in GL3PlusTextureGpu::copyTo and Gazebo aborts
    # -- so it falls to the software rasteriser, which measured 393.8 % of an
    # 8-core machine while the control loop got 50 %. Detaching it gives those
    # cores to the physics and the controller, and attaching it back costs
    # nothing but the cores again, for as long as you are watching.
    #
    # An earlier attempt at headless failed and the reason was elsewhere: '-r'
    # was still in this list, so the world ran during the six seconds the launch
    # takes and the vehicle was on the floor before the controller had a sample.
    # With the world paused until engagement that race is gone.
    headless = '-s ' if os.environ.get('UAM_GUI', '1') == '0' else ''
    gazeboLaunch = IncludeLaunchDescription(
        gazebo_rosPackageLaunch,
        launch_arguments={'gz_args': [f'-v -v4 {headless}empty.sdf'],
                          'on_exit_shutdown': 'true'}.items()
    )
    
    spawnModelNodeGazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', robotXacroName,
            '-topic', 'robot_description',
            # The link origin, not the aircraft: the airframe is built 2 m above
            # this frame, so the vehicle spawns at 7 m and the arm tip hangs at
            # about 6. At the 1.0 that was here the aircraft sat at 3 m and the
            # arm reached the ground, curled against it to [0, -pi, 3.03], and
            # the run would not engage at all -- the controller waits for the arm
            # to hang straight and it never could. Every joint was already on a
            # contact, so no torque commanded could move it and the reaction terms
            # were reading a plant that was pinned.
            # The aircraft ends up at 12 m and the arm tip at about 11, so a
            # transient can lose several metres and the arm still never touches
            # anything. 5.0 was enough only when nothing went wrong.
            '-z', '10.0'
        ],
        output='screen',
    )
    
    nodeRobotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robotDescription, 'use_sim_time': True}]
    )
    
    bridge_params = os.path.join(
        get_package_share_directory(namePackage),
        'parameters',
        'bridge_parameters.yaml'
    )

    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ],
        output='screen',
    )
    
    launchDescriptionObject = LaunchDescription()
    launchDescriptionObject.add_action(gazeboLaunch)
    launchDescriptionObject.add_action(spawnModelNodeGazebo)
    launchDescriptionObject.add_action(nodeRobotStatePublisher) 
    launchDescriptionObject.add_action(start_gazebo_ros_bridge_cmd)
    
    return launchDescriptionObject
