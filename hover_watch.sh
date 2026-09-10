#!/bin/bash
# Open-loop hover with the GUI up, so the climb can be watched directly.
source /opt/ros/humble/setup.bash
source /home/giorgio/uam_ws/install/setup.bash
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=/home/giorgio/uam_ws/install/hexacopter_control/lib:$IGN_GAZEBO_SYSTEM_PLUGIN_PATH
export LIBGL_ALWAYS_SOFTWARE=1
OMEGA=${1:-569.3}
HOLD=${2:-90}

ros2 launch hexacopter_sim gazebo_model.launch.py > /tmp/gz.log 2>&1 &
for i in $(seq 1 40); do
  sleep 2
  ign model --list 2>/dev/null | grep -q hexacopter_robot && { echo "model up after $((i*2))s"; break; }
done

# Gate on the clock, not on the model appearing. A server that spawned the model
# but is not stepping reports a frozen pose, which reads exactly like a vehicle
# that will not lift -- and is what made the previous run look like a failure.
read_sim_ms () {
  timeout 4 ign topic -e -t /world/empty/stats 2>/dev/null     | grep -A3 'sim_time' | grep -m1 'sec:' | grep -oE '[0-9]+' | head -1
}
T1=$(timeout 5 ign topic -e -t /world/empty/stats 2>/dev/null | grep -m1 -A2 'sim_time' | tr -d ' \n')
sleep 4
T2=$(timeout 5 ign topic -e -t /world/empty/stats 2>/dev/null | grep -m1 -A2 'sim_time' | tr -d ' \n')
echo "clock sample 1: $T1"
echo "clock sample 2: $T2"
if [ "$T1" = "$T2" ]; then
  echo 'CLOCK NOT ADVANCING -- simulation is not stepping. Aborting before commanding.'
  exit 1
fi
echo "servers running: $(pgrep -fc 'ign gazebo server')"

livepose () {
  timeout 5 ign topic -e -t /world/empty/dynamic_pose/info 2>/dev/null     | grep -A6 'name: "hexacopter_robot"' | grep -A4 position | grep -E '^ +[xyz]:' | tr -d ' \n' | head -c 90
}
echo "rest pose:  $(livepose)"
echo "commanding omega=$OMEGA rad/s for ${HOLD}s -- WATCH THE WINDOW"
timeout $HOLD ros2 topic pub -r 100 /rotor_speeds std_msgs/msg/Float64MultiArray   "{data: [$OMEGA,$OMEGA,$OMEGA,$OMEGA,$OMEGA,$OMEGA]}" >/dev/null 2>&1 &
PUB=$!
for k in 1 2 3; do sleep 15; echo "  +$((k*15))s: $(livepose)"; done
wait $PUB 2>/dev/null
echo 'window over; Gazebo left up'
