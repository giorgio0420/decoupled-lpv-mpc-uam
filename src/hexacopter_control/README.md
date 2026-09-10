# Hexacopter control — Fortress 6

This package targets Gazebo / Ignition Fortress 6 (`ignition-gazebo6`) and ROS 2 Humble.

## Verified rotor geometry in `robot_base`

| index | joint | x [m] | y [m] |
|---:|---|---:|---:|
| 1 | `joint_front_left_prop` | 0.000000 | 0.200000 |
| 2 | `joint_front_right_prop` | 0.173203 | 0.100000 |
| 3 | `joint_right_prop` | 0.173205 | -0.100000 |
| 4 | `joint_back_right_prop` | 0.000000 | -0.200000 |
| 5 | `joint_back_left_prop` | -0.173203 | -0.100000 |
| 6 | `joint_left_prop` | -0.173205 | 0.100000 |

The rotor plane is at `z = 2.021 m` in `robot_base`; every rotor is at radius 0.2 m.
The source Xacro already has correct joint names. Repair only the swapped child-link identifiers:

* `joint_back_right_prop` must have child `prop_back_right_respondable`.
* `joint_back_left_prop` must have child `prop_back_left_respondable`.

Rename the matching `link`, `collision`, and `visual` names together, keeping all origin values unchanged.

## Build and run

Copy this package to `~/uam_ws/src/`, then on the Ubuntu / ROS machine:

```bash
source /opt/ros/humble/setup.bash
cd ~/uam_ws
colcon build --packages-select hexacopter_control
source install/setup.bash
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="$IGN_GAZEBO_SYSTEM_PLUGIN_PATH:$PWD/install/hexacopter_control/lib"
ros2 run hexacopter_control allocation_node --ros-args --params-file src/hexacopter_control/config/hexacopter.yaml
```

Insert the contents of `model_plugin.sdf.inc` into the model SDF passed to Fortress. The plugin subscribes to `/rotor_speeds` (`std_msgs/msg/Float64MultiArray`, six non-negative rad/s), animates the six joints, and applies the resultant thrust and yaw drag torque to `robot_base`.

`allocation_node` accepts `geometry_msgs/msg/Wrench` on `/wrench_command`: `force.z = T`, `torque.{x,y,z} = [tau_x,tau_y,tau_z]`; it publishes `/rotor_speeds`.

The CW/CCW signs in code alternate around the hexagon but are deliberately marked provisional. Command one rotor at low speed, observe its positive joint rotation, and set `spin` so that the actual propeller produces the desired yaw reaction. Calibrate `kf` and `km` before enabling the MPC.
