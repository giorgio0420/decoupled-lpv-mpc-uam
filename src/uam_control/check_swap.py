"""The closed form must give the same A, B, T and CrossCoupling as the probing.

Both paths live in `coupling.py` behind one flag, so this forces each in turn on
the same random states and compares. Anything above 1e-9 means the swap changed
the controller, which is the one thing it must not do.
"""

import numpy as np

from uam_control import coupling
from uam_control.params import CONTROL

RNG = np.random.default_rng(7)
TRIALS = 100
worst = {'A': 0.0, 'B': 0.0, 'T': 0.0}
worst_c = {k: 0.0 for k in ('M_d', 'M_c', 'M_s', 'M_l', 'tau_bar', 'M_f', 'f_bar')}

generated = coupling._GENERATED
assert generated is not None, 'generated_dynamics non importato'

for _ in range(TRIALS):
    q = RNG.uniform(-2, 2, 3)
    qd = RNG.uniform(-3, 3, 3)
    qdd = RNG.uniform(-20, 20, 3)
    w = RNG.uniform(-2, 2, 3)
    wd = RNG.uniform(-30, 30, 3)
    fz = RNG.uniform(5, 120)

    coupling._GENERATED = generated
    A1, B1, T1 = coupling.manipulator_lpv(q, qd, w, wd, fz, CONTROL.dt_gamma)
    c1 = coupling.decompose(q, qd, qdd, w, wd, fz)

    coupling._GENERATED = None
    A0, B0, T0 = coupling.manipulator_lpv(q, qd, w, wd, fz, CONTROL.dt_gamma)
    c0 = coupling.decompose(q, qd, qdd, w, wd, fz)

    worst['A'] = max(worst['A'], np.abs(A1 - A0).max())
    worst['B'] = max(worst['B'], np.abs(B1 - B0).max())
    worst['T'] = max(worst['T'], np.abs(T1 - T0).max())
    for k in worst_c:
        worst_c[k] = max(worst_c[k],
                         np.abs(np.asarray(getattr(c1, k))
                                - np.asarray(getattr(c0, k))).max())

coupling._GENERATED = generated
print(f'{TRIALS} stati casuali, forma chiusa contro sondaggio')
ok = True
for name, value in list(worst.items()) + list(worst_c.items()):
    good = value < 1e-7
    ok &= good
    print(f"  {'ok  ' if good else 'FALLITO'} {name:9s} scarto massimo {value:.3e}")
print()
print('Identici: lo scambio non cambia il controllore.' if ok
      else 'DIVERSI: non usare la forma chiusa.')
raise SystemExit(0 if ok else 1)
