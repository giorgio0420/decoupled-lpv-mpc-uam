#!/bin/bash
# Measure observed vs predicted joint acceleration, in free fall.
#
# Free fall is the point of the setup. With the base accelerating at g, the arm
# is weightless in the base frame, which is exactly what newton_euler models
# with f_z = 0. The applied torque is then the only torque acting, so gravity
# cannot dominate it -- that was how three of the seven earlier attempts died.
# No world-fixing joint is needed and the XACRO is not touched.
#
# Ordering matters. The launch passes no -r, so the server comes up paused;
# `ros2 topic pub` needs a second or two to start publishing. Starting the
# publisher into the paused world and unpausing afterwards puts the torque on
# the joint from sim time zero, and leaves the whole 1.0 s of fall usable.

TAU=${1:-0.2}
LOG=/tmp/ratio.log

for i in $(seq 1 10); do
  pkill -9 -f '[i]gn gazebo' 2>/dev/null
  pkill -9 -f '[t]opic pub' 2>/dev/null
  sleep 1
  [ "$(pgrep -f '[i]gn gazebo server' 2>/dev/null | wc -l)" = "0" ] && break
done
echo "stale servers left: $(pgrep -f '[i]gn gazebo server' 2>/dev/null | wc -l)"

source /opt/ros/humble/setup.bash
source /home/giorgio/uam_ws/install/setup.bash
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=/home/giorgio/uam_ws/install/hexacopter_control/lib
export LIBGL_ALWAYS_SOFTWARE=1

ros2 launch hexacopter_sim gazebo_model.launch.py > $LOG 2>&1 &

for i in $(seq 1 30); do
  sleep 2
  ign model --list 2>/dev/null | grep -q hexacopter_robot && break
done
echo "model spawned: $(ign model --list 2>/dev/null | grep -c hexacopter_robot)"

echo "paused before: $(ign topic -e -t /world/empty/stats -n 1 2>/dev/null | grep -c 'paused: true')"

ros2 topic pub -r 200 /joint_torques std_msgs/msg/Float64MultiArray \
  "{data: [$TAU, 0.0, 0.0]}" >/dev/null 2>&1 &
PUB=$!
sleep 4  # let the publisher come up while the world is still frozen

ign service -s /world/empty/control \
  --reqtype ignition.msgs.WorldControl \
  --reptype ignition.msgs.Boolean \
  --timeout 3000 --req 'pause: false' >/dev/null 2>&1

sleep 3   # ~1.0 s of sim time at the observed 0.45 real-time factor, then impact
kill -9 $PUB 2>/dev/null
pkill -9 -f '[i]gn gazebo' 2>/dev/null

grep -a '\[ARMDYN\]' $LOG | sed 's/.*\[ARMDYN\]/[ARMDYN]/' > /tmp/armdyn.txt
echo "ARMDYN samples: $(wc -l < /tmp/armdyn.txt)"
head -2 /tmp/armdyn.txt
tail -1 /tmp/armdyn.txt
exit 0
