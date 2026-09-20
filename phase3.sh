#!/bin/bash
# Cleanup, then refuse to run unless exactly one world is up.
#
# The earlier version used 'pkill -9 -f "ign gazebo"'. With -f, pkill matches
# the whole command line, and that string was sitting in this script's own
# command line -- so it killed the shell before it reached Gazebo. Every failed
# run left a server alive; four had piled up, all publishing /uav/odom, and the
# controller was reading a blend of four worlds. Match on the process name with
# -x instead, and verify afterwards rather than assuming.
for i in $(seq 1 10); do
  pkill -9 -f '[i]gn gazebo' 2>/dev/null
  true
  sleep 1
  [ "$(pgrep -f '[i]gn gazebo server' 2>/dev/null | wc -l)" = "0" ] && break
done
LEFT=$(pgrep -f '[i]gn gazebo server' 2>/dev/null | wc -l)
if [ "$LEFT" != "0" ]; then echo "ABORT: $LEFT stale server(s) survived cleanup"; exit 1; fi

# Dynamically find the workspace root directory
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
source "${WS_DIR}/install/setup.bash"
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="${WS_DIR}/install/hexacopter_control/lib:${IGN_GAZEBO_SYSTEM_PLUGIN_PATH}"
# Table II's dt_eta (0.025 s) is simulate.py's default and it is right there --
# no wall clock to miss. Live here it is not: measured tick_ms median 25.5,
# p95 41.4, max 107.7 against the 25 ms period. 0.05 s is the rate this
# Python/OSQP controller can actually hold; override with UAM_DT_ETA=0.025 to
# reproduce the real-time miss for yourself.
export UAM_DT_ETA="${UAM_DT_ETA:-0.05}"
# Software rendering by default, hardware when UAM_GPU=1.
#
# This is not a choice about whether the window appears -- it appears either way.
# It decides who draws it. With the software rasteriser the GUI is drawn on the
# CPU, and measured with `top` during a run that costs 393.8 % of an 8-core
# machine while the controller gets 50 %. WSLg exposes the GPU here (/dev/dxg and
# libd3d12 are present), so the same picture can be drawn by the graphics card
# instead, leaving the cores to the physics and the control loop.
if [ "${UAM_GPU:-0}" = "1" ]; then
  unset LIBGL_ALWAYS_SOFTWARE
else
  export LIBGL_ALWAYS_SOFTWARE=1
fi
rm -f /tmp/phase3.log /tmp/ctrl.log
ros2 launch hexacopter_sim gazebo_model.launch.py > /tmp/phase3.log 2>&1 &
# Ask Gazebo what is up, rather than pattern-matching a process name. The GUI
# mode spawns a child whose command line contains "ign gazebo server"; headless
# (-s) has no such child, so a pgrep for that string read 0, the guard aborted,
# and its own cleanup killed the server it had started one line earlier. One
# model answering is also the condition that actually matters -- one world, one
# vehicle.
#
# Counted from the list this loop already fetched, not from a second call. The
# second call cost several seconds, and those seconds land between the model
# appearing and the loop closing: the arm folded to [0, -pi, 3.03] before the
# controller had a sample, and the run refused to engage at all.
for i in $(seq 1 30); do
  sleep 2
  MODELS=$(ign model --list 2>/dev/null)
  echo "$MODELS" | grep -q hexacopter_robot && break
done
N=$(echo "$MODELS" | grep -c hexacopter_robot)
if [ "$N" != "1" ]; then
  echo "ABORT: expected exactly 1 hexacopter_robot, found $N -- results would mix worlds"
  pkill -9 -f '[i]gn gazebo' 2>/dev/null; true
  exit 1
fi
echo "model up, count=$N"
sleep 0.3   # engage while the vehicle is still falling from its 5 m spawn, so the arm never reaches the ground and folds
# Start the controller while the world is still paused, then release it.
# Nothing moves until the loop is closed, so the arm is straight at q = 0 and
# every run begins from the same state.
# Two cores of its own, and a better scheduling priority.
#
# Not a micro-optimisation. Measured with `top` during a run on this 8-core
# machine, Gazebo took 393.8 % of CPU -- nearly four cores -- while the
# controller got 50 %, half of one. The same rotational MPC call costs 3.9 ms on
# an idle machine and 14.49 ms in flight: a factor of 3.7 that is contention, not
# Python. Two thirds of the control tick is the controller waiting for a core.
UAM_CPUS="${UAM_CPUS:-6,7}"
timeout ${1:-45} taskset -c "${UAM_CPUS}" nice -n -5 python3 "${WS_DIR}/src/uam_control/hover_node.py" --ros-args -p target_z:=2.0 -p arm_enabled:=${2:-true} > /tmp/ctrl.log 2>&1 &
CTRL=$!
sleep 6
ign service -s /world/empty/control --reqtype ignition.msgs.WorldControl \
  --reptype ignition.msgs.Boolean --timeout 3000 --req 'pause: false' >/dev/null 2>&1
echo "simulation released"
wait $CTRL 2>/dev/null
echo "modelli alla fine: $(ign model --list 2>/dev/null | grep -c hexacopter_robot)"
pkill -9 -f '[i]gn gazebo' 2>/dev/null; true
exit 0
