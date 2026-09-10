"""The manipulator on its own, with the vehicle held still.

The vehicle has been measured on its own (arm torques disabled: it tracks its
1 m cosine to about 1 m and does not diverge). The arm never has been. This
holds the base perfectly: omega = 0, omega_dot = 0, f_z = m g, which is what a
vehicle hovering rigidly would give the arm, and asks the same
`ManipulatorTubeMPC` that flies in Gazebo to follow the same joint reference.

Integration is 25 sub-steps of 1 ms per 25 ms control period, matching Gazebo's
physics rate. Stepping the plant once per control period is what made an
unstable controller look stable here once already.

Run: python3 arm_only.py [amplitude] [period]
"""

import sys

import numpy as np

from uam_control.controllers import ManipulatorTubeMPC
from uam_control.newton_euler import joint_accelerations
from uam_control.params import ARM, CONTROL, GRAVITY, LIMITS, TOTAL_MASS

AMPLITUDE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
PERIOD = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
# Disturbance amplitudes, from the Gazebo trace: body rate reached 2 rad/s, the
# rotational MPC's predicted rate derivative follows from +/-7 N m on I = 0.18,
# and thrust ran between 16 and 100 N against a 54 N hover.
OMEGA_AMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
OMEGA_DOT_AMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
THRUST_AMP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
DURATION = 20.0
SUBSTEPS = 25

zero = np.zeros(3)
hover = TOTAL_MASS * GRAVITY
dt = CONTROL.dt_gamma
sub = dt / SUBSTEPS

mpc = ManipulatorTubeMPC()
q = np.zeros(3)
qd = np.zeros(3)

errors = []
torques = []
accels = []
print(f"amplitude {AMPLITUDE} rad, period {PERIOD} s, "
      f"controller limit {LIMITS.joint_torque} N m")
print(f"{'t':>6} {'q':>28} {'qref':>8} {'|err|':>8} {'tau':>26}")
for step in range(int(DURATION / dt)):
    t = step * dt
    ref_angle = AMPLITUDE * np.sin(2 * np.pi * t / PERIOD)
    reference = np.tile(
        np.concatenate([np.full(3, ref_angle), zero]), (CONTROL.N_gamma, 1)
    )
    # The three things the vehicle hands the arm, driven at the amplitudes the
    # Gazebo trace actually recorded, so each can be switched on alone.
    wob = 2 * np.pi * t / 0.23  # the 0.23 s limit cycle the vehicle sits in
    omega = np.array([0.0, OMEGA_AMP * np.sin(wob), 0.0])
    omega_dot = np.array([0.0, OMEGA_DOT_AMP * np.cos(wob), 0.0])
    f_z = hover + THRUST_AMP * np.sin(wob)

    tau = mpc(np.concatenate([q, qd]), reference, omega, omega_dot, f_z)

    for _ in range(SUBSTEPS):
        qdd = joint_accelerations(q, qd, tau, omega, omega_dot, f_z)
        qd = np.clip(qd + sub * qdd, -LIMITS.joint_rate, LIMITS.joint_rate)
        q = q + sub * qd
        accels.append(np.abs(qdd).max())

    err = np.abs(q - ref_angle)
    errors.append(err.max())
    torques.append(np.abs(tau).max())
    if step % 40 == 0:
        print(f"{t:6.2f} {str(np.round(q, 3)):>28} {ref_angle:8.3f}"
              f" {err.max():8.3f} {str(np.round(tau, 2)):>26}")

errors = np.array(errors)
settled = errors[len(errors) // 2 :]
print()
print(f"peak |error|          {errors.max():.3f} rad")
print(f"settled |error|       median {np.median(settled):.3f}, max {settled.max():.3f} rad")
print(f"torque at the limit   {100 * np.mean(np.array(torques) > 0.99 * LIMITS.joint_torque):.1f}% of steps")
print(f"peak |qdd|            {max(accels):.1f} rad/s^2")

# The bench is only useful if it can fail. A tracker that never leaves the
# amplitude it was asked for has not tracked anything.
assert np.isfinite(errors).all(), "diverged to non-finite state"
