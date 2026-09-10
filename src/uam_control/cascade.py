"""Is the position-attitude-rate cascade stable, by calculation rather than by run?

Every Gazebo comparison in this project has a dispersion wide enough to swallow
the effect being measured -- identical configurations depart anywhere between 3
and 19 seconds. Eigenvalues have no dispersion.

The loop, one horizontal axis, taken from the code rather than from the paper:

    force_x  = m (kp (x_r - x) + kd (v_r - v))     PositionPD, bandwidth 0.8
    theta_r  = force_x / (m g)                     attitude_reference, small angle
    q_cmd    = ATTITUDE_GAIN (theta_r - theta)     hover_node
    tau_y    = K_eta (q_cmd - q)                   RotationalTubeMPC feedback term
    q_dot    = tau_y / I_yy
    theta_dot= q
    v_dot    = g theta

State [x, v, theta, q] plus one sample of delay on the torque, which the
implementation really has: the measurement that produces tau_y is one tick old.
Discretised at dt_eta with a zero-order hold, which is how it runs.
"""

import numpy as np
from scipy import linalg

from uam_control.coupling import decompose, rotational_lpv
from uam_control.mpc import lqr_gain
from uam_control.params import ARM, CONTROL, GRAVITY, LIMITS, TOTAL_MASS, UAV

g = GRAVITY
I = UAV.inertia_yy
m = UAV.mass
dt = CONTROL.dt_eta

# The rotational feedback gain the controller actually uses, at hover.
z3 = np.zeros(3)
coupling = decompose(z3, z3, z3, z3, z3, TOTAL_MASS * GRAVITY)
A_eta, B_eta = rotational_lpv(coupling, z3, dt)
K_eta = lqr_gain(A_eta, B_eta, CONTROL.Q_eta(), CONTROL.R_eta())
k_rate = abs(K_eta[1, 1])  # N m per rad/s of pitch-rate error


def spectral_radius(bandwidth, attitude_gain, delay=1):
    """Closed-loop spectral radius of the cascade, with `delay` samples of lag."""
    kp, kd = bandwidth ** 2, 2.0 * bandwidth

    # Continuous plant: x, v, theta, q driven by tau.
    Ac = np.zeros((4, 4))
    Ac[0, 1] = 1.0        # x_dot = v
    Ac[1, 2] = g          # v_dot = g theta
    Ac[2, 3] = 1.0        # theta_dot = q
    Bc = np.zeros((4, 1))
    Bc[3, 0] = 1.0 / I    # q_dot = tau / I

    n = 4
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    Md = linalg.expm(M * dt)
    Ad, Bd = Md[:n, :n], Md[:n, n:]

    # tau = k_rate (q_cmd - q), q_cmd = attitude_gain (theta_r - theta),
    # theta_r = (kp (0 - x) + kd (0 - v)) / g   [force / (m g), m cancels]
    theta_r = np.array([-kp / g, -kd / g, 0.0, 0.0])
    q_cmd = attitude_gain * (theta_r - np.array([0.0, 0.0, 1.0, 0.0]))
    K = k_rate * (q_cmd - np.array([0.0, 0.0, 0.0, 1.0]))  # row vector on the state

    if delay == 0:
        return max(abs(linalg.eigvals(Ad + Bd @ K.reshape(1, -1))))

    # One extra state holding the previous torque command.
    Aa = np.zeros((n + delay, n + delay))
    Aa[:n, :n] = Ad
    Aa[:n, n] = Bd[:, 0]
    for j in range(delay - 1):
        Aa[n + j + 1, n + j] = 1.0
    Ka = np.zeros(n + delay)
    Ka[:n] = K
    Aa[n, :] = Ka
    return max(abs(linalg.eigvals(Aa)))


print(f"guadagno di velocita angolare dal codice: K = {k_rate:.2f} N m per rad/s")
print(f"che su I_yy = {I} da {k_rate / I:.1f} rad/s^2 per rad/s,"
      f" contro 1/dt = {1 / dt:.0f}")
print()
print("raggio spettrale della cascata, dt = %.3f s" % dt)
print(f"{'banda':>7} {'ATT_GAIN':>9} {'senza ritardo':>14} {'1 campione':>12}"
      f" {'2 campioni':>12}")
for bw in (0.3, 0.5, 0.8, 1.0, 1.5):
    for ag in (0.5, 1.0, 2.0, 4.0):
        r0 = spectral_radius(bw, ag, delay=0)
        r1 = spectral_radius(bw, ag, delay=1)
        r2 = spectral_radius(bw, ag, delay=2)
        flag = ""
        if r1 > 1.0:
            flag = "   INSTABILE con 1 campione"
        print(f"{bw:7.1f} {ag:9.1f} {r0:14.4f} {r1:12.4f} {r2:12.4f}{flag}")
