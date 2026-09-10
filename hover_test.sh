#!/bin/bash
source /opt/ros/humble/setup.bash
source /home/giorgio/uam_ws/install/setup.bash
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=/home/giorgio/uam_ws/install/hexacopter_control/lib:$IGN_GAZEBO_SYSTEM_PLUGIN_PATH
export LIBGL_ALWAYS_SOFTWARE=1
OMEGA=${1:-569.3}
DUR=${2:-12}

ros2 launch hexacopter_sim gazebo_model.launch.py > /tmp/gz.log 2>&1 &
LAUNCH=$!
trap "kill -9 $LAUNCH 2>/dev/null; pkill -9 -f ign-gazebo 2>/dev/null; pkill -9 -f ruby 2>/dev/null" EXIT

for i in $(seq 1 40); do
  sleep 2
  if ign model --list 2>/dev/null | grep -q hexacopter_robot; then echo "model spawned after ${i}x2 s"; break; fi
done
ign model --list 2>/dev/null | head

echo "--- plugin load lines ---"
grep -iE "HexacopterRotor|Missing joint" /tmp/gz.log | head -10

echo "--- baseline pose ---"
ign model -m hexacopter_robot -p 2>/dev/null | head -4

echo "--- commanding omega=$OMEGA on all six rotors for ${DUR}s ---"
timeout $DUR ros2 topic pub -r 50 /rotor_speeds std_msgs/msg/Float64MultiArray \
  "{data: [$OMEGA,$OMEGA,$OMEGA,$OMEGA,$OMEGA,$OMEGA]}" > /dev/null 2>&1 &
PUB=$!
for t in 2 4 6 8 10; do
  sleep 2
  Z=$(ign model -m hexacopter_robot -p 2>/dev/null | grep -A3 "\[Pose\]" | head -4 | tail -1)
  echo "  t=${t}s  $(ign model -m hexacopter_robot -p 2>/dev/null | sed -n "/Pose/,+3p" | tr "\n" " " | head -c 160)"
done
kill $PUB 2>/dev/null
