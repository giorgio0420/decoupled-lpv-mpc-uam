# Tube-based LPV-MPC for an aerial manipulator

ROS 2 / Gazebo simulation of a hexarotor carrying a 3-link arm, after
Eskandarpour et al., *Decoupled Dynamic Modeling by Decomposing the
Cross-Coupled Dynamics and Tube-Based LPV-MPC Control Scheme for Aerial
Manipulation*, IEEE TAES 61(5), 2025. Tube-based LPV-MPC (3 MPCs:
translational, rotational, arm) against a PD + ERTF baseline, across two
scenarios (`nominal`, `fast`), visualized live in Gazebo.

## Run it

```bash
# once
sudo apt install -y ros-humble-desktop ros-humble-ros-gz \
  ros-humble-ros-gz-sim ros-humble-ros-gz-bridge \
  ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
  ros-humble-xacro python3-colcon-common-extensions python3-pip
pip install -r requirements.txt

git clone <repo URL> ~/uam_ws && cd ~/uam_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash

# fly a scenario (timeout seconds, arm enabled)
UAM_SCENARIO=nominal bash phase3.sh 95 true
```

Scenarios: `hover`, `nominal`, `fast`. Scheme switches:

| variable | default | what it does |
|---|---|---|
| `UAM_SCENARIO` | `nominal` | which reference to fly |
| `UAM_TRANS_MPC` | `0` | `1` = 3MPC (translational MPC instead of PD) |
| `UAM_USE_ERTF` | `0` | `1` = PD + ERTF baseline instead of the tube MPCs |

Output: `/tmp/trace.csv`. Read it with:

```bash
python3 tools/iae.py run=/tmp/trace.csv
python3 tools/report.py run=/tmp/trace.csv --out report
```

## Layout

```
phase3.sh                     launches Gazebo and the flight node for one run
src/hexacopter_sim/           XACRO model, launch file, ROS-Gazebo bridge
src/hexacopter_control/       Gazebo C++ plugin: rotor speeds -> forces/torques
src/uam_control/hover_node.py the flight node: cascade, allocation, tracing
src/uam_control/uam_control/  dynamics, controllers (3MPC / ERTF), trajectories
src/uam_control/derive_dynamics.py   regenerates generated_dynamics.py (SymPy)
tools/                        iae.py, report.py — read a trace.csv
```

WSL2 without a GPU: `phase3.sh` already forces software rendering
(`LIBGL_ALWAYS_SOFTWARE=1`); use `UAM_GUI=0` for real measurements, the GUI
alone costs ~4x the CPU of the control loop.
