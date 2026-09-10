"""Can the joints hold the arm at all, and how does thrust change the answer?

The manipulator's predicted joint acceleration runs at 200 rad/s^2, which is not
a physical number for a joint limited to 10 rad/s. Either the prediction is
broken or the arm genuinely cannot be held -- and the second is testable: the
torque needed to keep it still is one Newton-Euler evaluation at zero
acceleration. The base's pseudo-gravity scales with f_z / m, so the same question
gets a different answer at hover and at the 122 N peak the trace recorded.
"""

import numpy as np

from uam_control.newton_euler import newton_euler
from uam_control.params import ARM, GRAVITY, LIMITS, TOTAL_MASS

np.set_printoptions(precision=3, suppress=True)

rest = np.zeros(ARM.n_links)
hover = TOTAL_MASS * GRAVITY

poses = {
    "hanging, q = 0": np.zeros(3),
    "one joint out, q = [0.5, 0, 0]": np.array([0.5, 0.0, 0.0]),
    "as flown, q = [0.08, -1.73, -1.64]": np.array([0.08, -1.73, -1.64]),
    "worst case, arm horizontal": np.array([np.pi / 2, 0.0, 0.0]),
}

print(f"joint torque limit        +/- {LIMITS.joint_torque} N m")
print(f"hover thrust              {hover:.1f} N")
print(f"trace recorded thrust     18.2 to 122.0 N")
print()

for label, q in poses.items():
    print(label)
    for f_z, tag in [(hover, "hover"), (122.0, "thrust peak"), (18.2, "thrust trough")]:
        tau = newton_euler(q, rest, rest, np.zeros(3), np.zeros(3), f_z).joint_torques
        over = np.abs(tau) > LIMITS.joint_torque
        flag = "   EXCEEDS THE LIMIT" if np.any(over) else ""
        print(f"   {tag:14s} f_z = {f_z:6.1f} N   holding torque {np.round(tau, 3)}{flag}")
    print()
