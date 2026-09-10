"""Where does a run stop being good?

Short runs and long runs disagreed on the same configuration, which means the
short window ended before something happened. This walks the whole trace and
prints where the position error first leaves a threshold, plus a coarse timeline.
"""

import sys

import numpy as np

path = sys.argv[1]
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

d = np.genfromtxt(path, delimiter=",", names=True)
err = np.hypot(d["x"] - d["ref_x"], d["y"] - d["ref_y"])
bad = np.nonzero(err > threshold)[0]

print(f"{path}, {d['t'][-1]:.1f} s")
if bad.size:
    print(f"errore orizzontale supera {threshold} m per la prima volta a"
          f" t = {d['t'][bad[0]]:.2f} s")
else:
    print(f"errore orizzontale mai oltre {threshold} m")
print(f"errore finale {err[-1]:.2f} m, massimo {err.max():.2f} m")
print()
print(f"{'t':>7} {'x':>9} {'y':>9} {'z':>7} {'err':>8} {'pitch':>8}"
      f" {'thrust':>8} {'|q-qref|':>9} {'sfit':>6}")
for tt in range(0, int(d["t"][-1]) + 1, 5):
    r = d[np.argmin(np.abs(d["t"] - tt))]
    qe = max(abs(r["q0"] - r["qref0"]), abs(r["q1"] - r["qref1"]),
             abs(r["q2"] - r["qref2"]))
    e = np.hypot(r["x"] - r["ref_x"], r["y"] - r["ref_y"])
    print(f"{r['t']:7.1f} {r['x']:9.2f} {r['y']:9.2f} {r['z']:7.2f} {e:8.2f}"
          f" {r['pitch']:8.3f} {r['thrust']:8.1f} {qe:9.3f} {r['sfit']:6.3f}")
