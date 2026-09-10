"""Does RotationalTubeMPC behave like its LQR gain?

The eigenvalue analysis assumed tau = K (q_cmd - q): the gain acting on the
tracking error. Reading the code, that is not what it computes. It computes

    error       = body_rate - nominal_state          (tube deviation, not tracking)
    total_input = K @ error + nominal_input          (QP does the tracking)
    torque      = clip(total_input - tau_bar)

So K corrects departures from the nominal trajectory while the QP tracks the
reference, and the effective tracking law is whatever the QP produces. This
drives the real controller in a closed loop against the real rate dynamics, with
the one sample of actuation delay the implementation has, and reads the answer
off the response instead of assuming it.
"""

import numpy as np

from uam_control.controllers import RotationalTubeMPC
from uam_control.coupling import decompose, rotational_lpv
from uam_control.mpc import lqr_gain
from uam_control.params import CONTROL, GRAVITY, TOTAL_MASS, UAV

I = np.diag(UAV.inertia_tensor())
dt = CONTROL.dt_eta
z3 = np.zeros(3)
hover = TOTAL_MASS * GRAVITY
coupling = decompose(z3, z3, z3, z3, z3, hover)

A_eta, B_eta = rotational_lpv(coupling, z3, dt)
K = lqr_gain(A_eta, B_eta, CONTROL.Q_eta(), CONTROL.R_eta())
print(f"guadagno LQR sul canale di beccheggio: K_yy = {K[1,1]:.3f} N m per rad/s")
print(f"R_eta = {CONTROL.R_eta_gain}, dt = {dt} s, I_yy = {I[1]}")
print()

TARGET = 0.10  # rad/s di riferimento sulla velocita angolare di beccheggio
STEPS = 40

rot = RotationalTubeMPC()
rate = np.zeros(3)
tau_applied = np.zeros(3)   # un campione di ritardo, come nell impianto
reference = np.tile(np.array([0.0, TARGET, 0.0]), (CONTROL.N_eta, 1))

print("passo   q [rad/s]   tau comandata   K*(q_ref-q)   nominale   errore tubo")
rows = []
for k in range(STEPS):
    # Il ritardo: si integra con la coppia decisa al passo precedente.
    rate = rate + dt * tau_applied / I
    tau = rot(rate, reference, coupling)
    tracking = K[1, 1] * (TARGET - rate[1])
    nom = rot.nominal_state[1]
    tube_err = rate[1] - nom
    rows.append((rate[1], tau[1], tracking, nom, tube_err))
    if k < 16 or k % 6 == 0:
        print("%5d %11.5f %15.4f %13.4f %10.5f %13.5f"
              % (k, rate[1], tau[1], tracking, nom, tube_err))
    tau_applied = tau

r = np.array(rows)
print()
print(f"riferimento {TARGET} rad/s; velocita raggiunta dopo {STEPS} passi:"
      f" {r[-1, 0]:+.5f} rad/s")
flips = np.sum(np.diff(np.sign(r[:, 1])) != 0)
print(f"cambi di segno della coppia: {flips} su {STEPS} passi")
print(f"ampiezza della coppia: da {r[:, 1].min():+.3f} a {r[:, 1].max():+.3f} N m")
print()
if np.abs(r[-8:, 0]).max() > 5 * abs(TARGET):
    print("L ANELLO DIVERGE con il solo ritardo di un campione, senza impianto,")
    print("senza braccio e senza Gazebo.")
elif abs(r[-1, 0] - TARGET) < 0.1 * abs(TARGET):
    print("L anello insegue il riferimento.")
else:
    print("L anello non diverge ma non insegue: errore residuo "
          f"{r[-1, 0] - TARGET:+.4f} rad/s")
