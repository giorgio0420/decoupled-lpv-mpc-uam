#!/bin/bash
# Open loop: hover thrust plus a fixed pitch torque. Which way does it go?
#
# The one link in the chain never checked in Gazebo. Every stage has been verified
# in isolation -- the arm model to 3%, attitude_reference exactly, the rotational
# loop tracks, the allocation delivers to 0.05 N m -- and the whole thing still
# oscillates in horizontal velocity alone, growing, at 0.25 s. Positive feedback
# around a fast inner loop looks exactly like that, and it needs only one sign to
# be wrong.
#
# Expected: tau_y > 0 tilts the thrust toward +x, so pitch rises and x rises with
# it. No controller runs here, so the vehicle tips over eventually; only the first
# fraction of a second matters.
# set -u tolto: il setup di ROS usa variabili non definite

TAU=${1:-0.3}     # [N m] about body y
HOLD=${2:-6}      # [s] wall clock to hold the command

for i in $(seq 1 10); do
  pkill -9 -f '[i]gn gazebo' 2>/dev/null
  pkill -9 -f '[t]opic echo' 2>/dev/null
  pkill -9 -f '[t]opic pub' 2>/dev/null
  sleep 1
  [ "$(pgrep -f '[i]gn gazebo server' 2>/dev/null | wc -l)" = "0" ] && break
done

source /opt/ros/humble/setup.bash
source "$HOME/uam_ws/install/setup.bash"
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="$HOME/uam_ws/install/hexacopter_control/lib"
export LIBGL_ALWAYS_SOFTWARE=1

# Rotor speeds for hover plus the pitch torque, from the same allocation the
# controller uses.
OM=$(cd "$HOME/uam_ws/uam_control" && python3 - "$TAU" <<'PYEOF'
import sys
import numpy as np
from uam_control.params import ROTORS, TOTAL_MASS, GRAVITY
tau = float(sys.argv[1])
alloc = ROTORS.allocation_matrix()
wrench = np.array([TOTAL_MASS * GRAVITY, 0.0, tau, 0.0])
squared = np.linalg.pinv(alloc) @ wrench
omega = np.sqrt(np.clip(squared, 0.0, ROTORS.max_omega ** 2))
print(",".join("%.4f" % w for w in omega))
PYEOF
)
echo "tau_y = $TAU N m  ->  omega = [$OM]"

ros2 launch hexacopter_sim gazebo_model.launch.py > /tmp/sign.log 2>&1 &
for i in $(seq 1 30); do
  sleep 2
  ign model --list 2>/dev/null | grep -q hexacopter_robot && break
done
echo "modello su: $(ign model --list 2>/dev/null | grep -c hexacopter_robot)"

# Publisher and recorder start while the world is still frozen, so the command is
# on the rotors from simulated time zero and nothing is missed at the start.
ros2 topic pub -r 100 /rotor_speeds std_msgs/msg/Float64MultiArray \
  "{data: [$OM]}" > /dev/null 2>&1 &
PUB=$!
ros2 topic echo /uav/odom --csv > /tmp/odom.csv 2>/dev/null &
REC=$!
sleep 4

ign service -s /world/empty/control --reqtype ignition.msgs.WorldControl \
  --reptype ignition.msgs.Boolean --timeout 3000 --req 'pause: false' > /dev/null 2>&1
echo "rilasciata"
sleep "$HOLD"

kill -9 $PUB $REC 2>/dev/null
pkill -9 -f '[i]gn gazebo' 2>/dev/null
echo "campioni di odometria: $(wc -l < /tmp/odom.csv)"
