"""The first tick where the run stops being right, and what the state was just before.

Every diagnosis so far has looked at the wreckage. This finds the onset: the first
sample whose horizontal error passes a small threshold, and prints the ticks either
side of it so the trigger is visible rather than inferred.
"""

import sys

import numpy as np

path = sys.argv[1]
thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

d = np.genfromtxt(path, delimiter=",", names=True)
err = np.hypot(d["x"] - d["ref_x"], d["y"] - d["ref_y"])
bad = np.nonzero(err > thresh)[0]
if not bad.size:
    print(f"{path}: errore mai oltre {thresh} m in {d['t'][-1]:.1f} s")
    raise SystemExit

i = bad[0]
print(f"{path}: primo superamento di {thresh} m a t = {d['t'][i]:.3f} s "
      f"(tick {i} di {d.size})")
print()
cols = ["t", "x", "y", "z", "pitch", "pitch_cmd", "q_cmd", "q_meas", "tau_y",
        "thrust", "fbar_x", "force_x", "q0", "q1", "q2", "sfit"]
print(" ".join(f"{c:>8s}" for c in cols))
lo, hi = max(0, i - 12), min(d.size, i + 12)
for r in d[lo:hi]:
    print(" ".join(f"{r[c]:8.3f}" for c in cols))
