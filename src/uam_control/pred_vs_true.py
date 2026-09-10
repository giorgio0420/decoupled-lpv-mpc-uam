"""Does the LPV one-step prediction agree with the forward dynamics?

`ManipulatorTubeMPC` reports its predicted joint acceleration as

    ((A @ x + B @ u)[n:] - qd) / dt,      u = tau + T

and that number is what drives the arm reaction correction the vehicle's
attitude loop subtracts. The trace has it at a median 198 rad/s^2 on a joint
limited to 10 rad/s, while the torque needed to hold the arm still is under
6 N m everywhere, so the arm is not being overpowered. This compares the LPV
prediction against `joint_accelerations`, which solves the same dynamics without
the linearisation, for the same state and the same applied torque.
"""

import numpy as np

from uam_control.coupling import manipulator_lpv
from uam_control.newton_euler import joint_accelerations
from uam_control.params import ARM, CONTROL, GRAVITY, TOTAL_MASS

np.set_printoptions(precision=3, suppress=True)

dt = CONTROL.dt_gamma
zero = np.zeros(3)
hover = TOTAL_MASS * GRAVITY

cases = [
    ("hanging, no torque", np.zeros(3), np.zeros(3), np.zeros(3)),
    ("hanging, 1 N m on joint 1", np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0])),
    ("as flown, no torque", np.array([0.08, -1.73, -1.64]), np.zeros(3), np.zeros(3)),
    ("as flown, moving", np.array([0.08, -1.73, -1.64]), np.array([1.0, -1.0, 1.0]), np.array([2.0, 1.0, 0.5])),
]

print(f"{'case':32s} {'LPV prediction':>26s} {'forward dynamics':>26s}")
for label, q, qd, tau in cases:
    A, B, residual = manipulator_lpv(q, qd, zero, zero, hover, dt)
    x = np.concatenate([q, qd])
    u = tau + residual
    pred = ((A @ x + B @ u)[3:] - qd) / dt
    true = joint_accelerations(q, qd, tau, zero, zero, hover)
    print(f"{label:32s} {str(np.round(pred, 2)):>26s} {str(np.round(true, 2)):>26s}")

print()
A, B, residual = manipulator_lpv(np.zeros(3), np.zeros(3), zero, zero, hover, dt)
print(f"T (residual) with the arm hanging at rest:  {np.round(residual, 3)} N m")
print("The arm hangs in equilibrium there, so the torque that holds it is zero")
print("and T should be zero with it.")
