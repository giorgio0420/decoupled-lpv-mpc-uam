"""The manipulator against Section V's own reference, with the vehicle held rigid.

This is the baseline the whole question rests on: if the MPC cannot follow the
paper's joint reference with a perfectly still base, nothing done to the vehicle
will help. The base is fixed exactly as a rigidly hovering one would be --
omega = 0, omega_dot = 0, f_z = m g -- and the reference is `trajectory.py`'s
square wave, the same one that flies in Gazebo.

Integration is 25 sub-steps of 1 ms per 25 ms control period, matching Gazebo.
Stepping the plant once per control period made an unstable controller look
stable here once already.

    python3 arm_paper.py [scenario] [omega_dot] [horizon]

`omega_dot` injects the vehicle's angular acceleration at the amplitude the
Gazebo trace recorded, so the rigid-base result and the flying one can be
compared with one variable between them. `horizon` overrides N_gamma, to ask
what preview buys against a reference that steps every 5 s.
"""

import sys

import numpy as np

from uam_control.controllers import ManipulatorTubeMPC
from uam_control.newton_euler import joint_accelerations
from uam_control.params import ARM, CONTROL, GRAVITY, LIMITS, TOTAL_MASS
from uam_control.trajectory import SCENARIOS, horizon_reference, joint_reference

NAME = sys.argv[1] if len(sys.argv) > 1 else "nominal"
OMEGA_DOT_AMP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
HORIZON = int(sys.argv[3]) if len(sys.argv) > 3 else CONTROL.N_gamma

scenario = SCENARIOS[NAME]
zero = np.zeros(3)
hover = TOTAL_MASS * GRAVITY
dt = CONTROL.dt_gamma
sub = dt / 25
DURATION = 30.0

mpc = ManipulatorTubeMPC()
mpc.horizon = HORIZON
q = np.zeros(3)
qd = np.zeros(3)

rows = []
print(f"scenario '{NAME}': gradini a {scenario.joint_amplitude} rad, "
      f"periodo {scenario.joint_period} s, partenza a {scenario.joint_start} s")
print(f"limite di coppia {LIMITS.joint_torque} N m, orizzonte {HORIZON} passi "
      f"= {HORIZON * dt * 1000:.0f} ms, omega_dot iniettata {OMEGA_DOT_AMP} rad/s^2")
print()
print(f"{'t':>6} {'q':>26} {'qref':>8} {'|err|':>8} {'tau':>24}")
for step in range(int(DURATION / dt)):
    t = step * dt
    reference = horizon_reference(joint_reference, scenario, t, HORIZON, dt)
    ref_now = joint_reference(scenario, t)[:3]

    # The vehicle's angular acceleration, at the 0.23 s period of the limit cycle
    # the two bodies sit in when flying.
    wob = 2 * np.pi * t / 0.23
    omega_dot = np.array([0.0, OMEGA_DOT_AMP * np.cos(wob), 0.0])

    tau = mpc(np.concatenate([q, qd]), reference, zero, omega_dot, hover)
    for _ in range(25):
        qdd = joint_accelerations(q, qd, tau, zero, omega_dot, hover)
        qd = np.clip(qd + sub * qdd, -LIMITS.joint_rate, LIMITS.joint_rate)
        q = q + sub * qd

    err = np.abs(q - ref_now)
    rows.append((t, err.max(), np.abs(tau).max()))
    if step % 60 == 0:
        print(f"{t:6.2f} {str(np.round(q, 3)):>26} {ref_now[0]:8.3f}"
              f" {err.max():8.3f} {str(np.round(tau, 2)):>24}")

rows = np.array(rows)
after = rows[rows[:, 0] > scenario.joint_start + 1.0]
print()
print(f"errore |q - qref| dopo l'avvio dei giunti:"
      f"  mediano {np.median(after[:, 1]):.3f}"
      f"  medio {after[:, 1].mean():.3f}"
      f"  massimo {after[:, 1].max():.3f} rad")
print(f"riferimento di ampiezza {scenario.joint_amplitude} rad, quindi l'errore"
      f" e' il {100 * np.median(after[:, 1]) / scenario.joint_amplitude:.0f}%"
      f" dell'ampiezza")
print(f"coppia al limite sul {100 * np.mean(after[:, 2] > 0.99 * LIMITS.joint_torque):.1f}%"
      f" dei passi")
